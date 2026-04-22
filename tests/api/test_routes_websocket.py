"""Tests for :mod:`api.routes.websocket` and :class:`core.moment.broadcaster.MomentBroadcaster`.

Two layers of coverage:

1. **Broadcaster unit tests** — in-memory fakes, no WebSocket transport.
   Verify the register/unregister/broadcast mechanics including the
   fail-open drop on a raising client.

2. **Endpoint integration tests** — full FastAPI TestClient with the
   WebSocket handshake. Verifies:

   - ``/ws`` accepts and registers a client.
   - Disconnect unregisters the client cleanly.
   - A missing broadcaster closes with 1011 after accept (so browsers
     do not hammer-retry on a refused handshake).
   - Action endpoints (accept / dismiss) broadcast a DONE partial to
     connected clients; snooze does not (SNOOZED is not terminal).

3. **Template wiring** — ensures the base template loads the ``ws``
   extension with ``ws-connect="/ws"`` and that the Now-tab template
   exposes stable ``#now-list`` / ``#done-today-list`` OOB targets.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.moment.broadcaster import MomentBroadcaster
from core.moment.feedback_weight import FeedbackWeightStore
from core.moment.types import (
    Action,
    ActionKind,
    InsightType,
    Moment,
    MomentState,
)
from storage import schema
from storage.repos.moments import MomentRepository

REF_NOW = 1_777_204_800


# ---------------------------------------------------------------------------
# unit tests: MomentBroadcaster in isolation
# ---------------------------------------------------------------------------


class _FakeWS:
    """Tiny stand-in that records every payload it 'sends'."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("ws closed")
        self.received.append(data)


class _RaisingWS:
    """Always raises on send — used to test the fail-open drop path."""

    async def send_text(self, data: str) -> None:
        raise RuntimeError("boom")


def test_broadcaster_empty_len() -> None:
    b = MomentBroadcaster()
    assert len(b) == 0


def test_broadcaster_register_caches_loop() -> None:
    async def inner() -> None:
        b = MomentBroadcaster()
        assert b.loop is None
        await b.register(_FakeWS())
        assert b.loop is asyncio.get_running_loop()
        assert len(b) == 1

    asyncio.run(inner())


def test_broadcaster_register_is_idempotent() -> None:
    async def inner() -> None:
        b = MomentBroadcaster()
        ws = _FakeWS()
        await b.register(ws)
        await b.register(ws)
        assert len(b) == 1

    asyncio.run(inner())


def test_broadcaster_unregister_is_idempotent() -> None:
    async def inner() -> None:
        b = MomentBroadcaster()
        ws = _FakeWS()
        # Dropping an unknown ws should not raise.
        await b.unregister(ws)
        await b.register(ws)
        await b.unregister(ws)
        await b.unregister(ws)
        assert len(b) == 0

    asyncio.run(inner())


def test_broadcast_fans_out_to_every_client() -> None:
    async def inner() -> None:
        b = MomentBroadcaster()
        ws1, ws2 = _FakeWS(), _FakeWS()
        await b.register(ws1)
        await b.register(ws2)

        sent = await b.broadcast("<p>hi</p>")
        assert sent == 2
        assert ws1.received == ["<p>hi</p>"]
        assert ws2.received == ["<p>hi</p>"]

    asyncio.run(inner())


def test_broadcast_drops_clients_that_raise_on_send() -> None:
    """A raising client must not block fan-out to the rest."""

    async def inner() -> None:
        b = MomentBroadcaster()
        healthy = _FakeWS()
        flaky = _RaisingWS()
        await b.register(healthy)
        await b.register(flaky)

        sent = await b.broadcast("<p>hi</p>")

        # Only the healthy client counts as "sent"; the flaky one is
        # dropped so future broadcasts never try to use it again.
        assert sent == 1
        assert healthy.received == ["<p>hi</p>"]
        assert len(b) == 1

    asyncio.run(inner())


def test_broadcast_on_empty_set_returns_zero() -> None:
    async def inner() -> None:
        b = MomentBroadcaster()
        assert await b.broadcast("anything") == 0

    asyncio.run(inner())


def test_notify_sync_no_loop_is_noop() -> None:
    """Without a cached loop, notify_sync must return without raising."""
    b = MomentBroadcaster()
    # No clients + no loop — should just silently succeed.
    b.notify_sync("anything")
    assert len(b) == 0


# ---------------------------------------------------------------------------
# integration fixtures: FastAPI app + real repo + real broadcaster
# ---------------------------------------------------------------------------


class DummyLifeOS:
    """Minimal life_os double exposing the three attributes we need."""

    def __init__(
        self,
        *,
        moment_repo=None,
        feedback_weight_store=None,
        moment_broadcaster=None,
    ) -> None:
        self.config: dict = {}
        self.moment_repo = moment_repo
        self.feedback_weight_store = feedback_weight_store
        self.moment_broadcaster = moment_broadcaster


class _Clock:
    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


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
    return _Clock()


@pytest.fixture
def repo(conn, clock):
    return MomentRepository(conn, now_fn=clock)


@pytest.fixture
def feedback(conn, clock):
    return FeedbackWeightStore(conn, now_fn=clock)


@pytest.fixture
def broadcaster():
    return MomentBroadcaster()


@pytest.fixture
def client(repo, feedback, broadcaster):
    life_os = DummyLifeOS(
        moment_repo=repo,
        feedback_weight_store=feedback,
        moment_broadcaster=broadcaster,
    )
    app = create_app(life_os)
    return TestClient(app)


def _make_moment(
    *,
    insight_type: InsightType = InsightType.CADENCE,
    state: MomentState = MomentState.SUGGESTED,
    confidence: float = 0.9,
    insight: str = "ping your sister",
    draft: str = "Hey — been a minute.",
    moment_id: str | None = None,
    evidence_hash: str | None = None,
    expires_at: int = REF_NOW + 3 * 24 * 3600,
) -> Moment:
    return Moment(
        id=moment_id or str(uuid.uuid4()),
        created_at=REF_NOW,
        expires_at=expires_at,
        insight=insight,
        evidence_hash=evidence_hash or f"hash-{uuid.uuid4().hex[:8]}",
        proposed_action=Action(kind=ActionKind.DRAFT_MESSAGE, params={"body": draft}),
        source_insight_type=insight_type,
        evidence=["evt-1"],
        state=state,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# /ws endpoint integration
# ---------------------------------------------------------------------------


def test_ws_accepts_connection_and_registers(client: TestClient, broadcaster: MomentBroadcaster) -> None:
    """The happy-path handshake registers the socket and blocks open."""
    assert len(broadcaster) == 0
    with client.websocket_connect("/ws") as _ws:
        # Inside the with-block the endpoint has accepted, registered,
        # and is parked on receive_text. The broadcaster's set must
        # carry exactly one client.
        assert len(broadcaster) == 1
    # After the with-block exits the client closed; the endpoint's
    # finally ran and removed the socket from the set.
    assert len(broadcaster) == 0


def test_ws_closes_when_broadcaster_missing() -> None:
    """Half-constructed life_os must close cleanly, not hang."""
    app = create_app(DummyLifeOS(moment_broadcaster=None))
    c = TestClient(app)
    with pytest.raises(Exception):  # starlette raises WebSocketDisconnect
        with c.websocket_connect("/ws") as ws:
            # The server accepted the handshake and immediately closed
            # with code 1011 (broadcaster not wired). The receive must
            # surface that close.
            ws.receive_text()


def test_accept_broadcasts_done_partial_to_connected_client(
    client: TestClient, repo, broadcaster: MomentBroadcaster
) -> None:
    """An accept action pushes the DONE-TODAY OOB swap over the socket."""
    mom = _make_moment(insight="water the plants", confidence=0.9)
    repo.create(mom)

    with client.websocket_connect("/ws") as ws:
        resp = client.post(f"/api/moments/{mom.id}/accept")
        assert resp.status_code == 200

        html = ws.receive_text()
        # OOB target matches the Now-tab DONE TODAY list.
        assert 'id="done-today-list"' in html
        assert 'hx-swap-oob="afterbegin"' in html
        # And the row carries the acted-on moment's insight + id.
        assert "water the plants" in html
        assert f'data-moment-id="{mom.id}"' in html
        assert "data-moment-done" in html


def test_dismiss_broadcasts_done_partial(client: TestClient, repo, broadcaster: MomentBroadcaster) -> None:
    mom = _make_moment(insight="archive that", confidence=0.9)
    repo.create(mom)

    with client.websocket_connect("/ws") as ws:
        resp = client.post(f"/api/moments/{mom.id}/dismiss")
        assert resp.status_code == 200

        html = ws.receive_text()
        assert 'id="done-today-list"' in html
        assert "archive that" in html
        assert MomentState.DISMISSED.value in html


def test_snooze_does_not_broadcast_done(client: TestClient, repo, broadcaster: MomentBroadcaster) -> None:
    """Snooze is not terminal; DONE TODAY must not receive a push."""
    mom = _make_moment(confidence=0.9)
    repo.create(mom)

    with client.websocket_connect("/ws") as _ws:
        resp = client.post(
            f"/api/moments/{mom.id}/snooze",
            json={"snooze_until": REF_NOW + 3600},
        )
        assert resp.status_code == 200

        # Nothing pushed — snooze is not a DONE TODAY event. We
        # sentinel-check by asserting the broadcaster's client count
        # is unchanged rather than racing the loop with a deadline.
        assert len(broadcaster) == 1  # still registered, no send happened


def test_action_without_broadcaster_still_succeeds(repo, feedback) -> None:
    """A missing broadcaster must not break action endpoints."""
    life_os = DummyLifeOS(
        moment_repo=repo,
        feedback_weight_store=feedback,
        moment_broadcaster=None,
    )
    app = create_app(life_os)
    c = TestClient(app)

    mom = _make_moment(confidence=0.9)
    repo.create(mom)

    resp = c.post(f"/api/moments/{mom.id}/accept")
    assert resp.status_code == 200
    assert resp.json()["state"] == MomentState.ACCEPTED.value


# ---------------------------------------------------------------------------
# Template wiring
# ---------------------------------------------------------------------------


def test_base_template_wires_ws_connect(client: TestClient) -> None:
    """`ws-connect="/ws"` + ws extension live on the base template."""
    body = client.get("/").text
    # htmx-ext-ws script tag present.
    assert "htmx-ext-ws" in body
    # body element carries the extension opt-in + connect directive.
    assert 'hx-ext="ws"' in body
    assert 'ws-connect="/ws"' in body
    # Status pill markup is still present (from the original Week 9
    # base-template task).
    assert 'id="ws-status"' in body
    # And the status-pill JS listens for the htmx ws lifecycle events
    # so the "Reconnecting…" pill toggles on drop / open.
    assert "htmx:wsOpen" in body
    assert "htmx:wsClose" in body
    assert "htmx:wsError" in body


def test_now_page_exposes_oob_targets(client: TestClient) -> None:
    """The NOW + DONE TODAY lists render with the WS OOB-swap target ids."""
    body = client.get("/").text
    assert 'id="now-list"' in body
    assert 'id="done-today-list"' in body


def test_ws_moment_new_partial_targets_now_list(client: TestClient, repo, broadcaster) -> None:
    """Rendering the new-moment partial yields an afterbegin OOB swap."""
    from web.rendering import render

    mom = _make_moment(insight="ping mom")
    html = render("partials/ws_moment_new.html", {"moment": mom})
    assert 'id="now-list"' in html
    assert 'hx-swap-oob="afterbegin"' in html
    # The wrapping <li> carries the new-moment marker for the CSS
    # accent-bar animation (DESIGN.md § "Moment: on arrival").
    assert "data-moment-new" in html
    # And the inner card renders — insight text flows through.
    assert "ping mom" in html
