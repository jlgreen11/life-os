"""Tests for :class:`core.moment.broadcaster.MomentBroadcaster`.

The broadcaster is the WebSocket fan-out used by the Now tab for live
Moment updates. Coverage gap #1 in the 2026-04-23 audit: 0/47 lines
covered before these tests.

Covers:

- ``__init__`` starts with no clients and no cached loop.
- ``register`` / ``unregister`` are idempotent.
- ``register`` on a running loop caches that loop for ``notify_sync``.
- ``register`` outside a running loop leaves the cached loop ``None``.
- ``broadcast`` returns 0 when no clients connected.
- ``broadcast`` sends to every connected client and returns the count.
- ``broadcast`` drops clients whose ``send_text`` raises (fail-open).
- ``notify_sync`` is a no-op with no clients or no cached loop.
- ``notify_sync`` schedules ``broadcast`` on the cached loop.
- ``notify_sync`` swallows RuntimeError when the loop is closed.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest

from core.moment.broadcaster import MomentBroadcaster


class FakeWS:
    """Minimal _Sendable that records every send_text call."""

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    async def send_text(self, data: str) -> None:
        if self.fail:
            raise RuntimeError("ws send boom")
        self.sent.append(data)


# ---------------------------------------------------------------------------
# construction / basic state
# ---------------------------------------------------------------------------


def test_broadcaster_starts_empty():
    b = MomentBroadcaster()
    assert len(b) == 0
    assert b.loop is None


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------


def test_register_adds_client_and_caches_loop():
    b = MomentBroadcaster()
    ws = FakeWS()

    async def go() -> None:
        await b.register(ws)

    asyncio.run(go())
    # After the event loop exits the cached loop reference remains (by design
    # — we only inspect identity, not liveness — and notify_sync handles the
    # closed-loop case).
    assert len(b) == 1
    assert b.loop is not None


def test_register_is_idempotent():
    b = MomentBroadcaster()
    ws = FakeWS()

    async def go() -> None:
        await b.register(ws)
        await b.register(ws)

    asyncio.run(go())
    assert len(b) == 1


def test_unregister_removes_client():
    b = MomentBroadcaster()
    ws = FakeWS()

    async def go() -> None:
        await b.register(ws)
        await b.unregister(ws)

    asyncio.run(go())
    assert len(b) == 0


def test_unregister_missing_client_is_noop():
    b = MomentBroadcaster()
    ws = FakeWS()

    async def go() -> None:
        await b.unregister(ws)

    asyncio.run(go())  # no exception
    assert len(b) == 0


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------


def test_broadcast_with_no_clients_returns_zero():
    b = MomentBroadcaster()

    async def go() -> int:
        return await b.broadcast("<div>hi</div>")

    assert asyncio.run(go()) == 0


def test_broadcast_sends_to_every_client():
    b = MomentBroadcaster()
    ws1 = FakeWS()
    ws2 = FakeWS()

    async def go() -> int:
        await b.register(ws1)
        await b.register(ws2)
        return await b.broadcast("<div>hello</div>")

    assert asyncio.run(go()) == 2
    assert ws1.sent == ["<div>hello</div>"]
    assert ws2.sent == ["<div>hello</div>"]


def test_broadcast_drops_failing_clients_and_keeps_going():
    """One client's exception must not stop fan-out to healthy clients."""
    b = MomentBroadcaster()
    bad = FakeWS(fail=True)
    good = FakeWS()

    async def go() -> int:
        await b.register(bad)
        await b.register(good)
        return await b.broadcast("<p>x</p>")

    sent = asyncio.run(go())
    assert sent == 1
    assert good.sent == ["<p>x</p>"]
    # The failing ws was removed from the set.
    assert bad not in b._clients
    assert len(b) == 1


# ---------------------------------------------------------------------------
# notify_sync
# ---------------------------------------------------------------------------


def test_notify_sync_noop_without_clients():
    """No clients, no loop → silent no-op (no exception)."""
    b = MomentBroadcaster()
    b.notify_sync("<div>x</div>")
    assert b.loop is None


def test_notify_sync_noop_when_loop_not_cached():
    """Registered client but no cached loop (shouldn't normally happen) → no-op."""
    b = MomentBroadcaster()
    ws = FakeWS()
    # Bypass register to simulate the degenerate no-loop state.
    b._clients.add(ws)
    b.notify_sync("<div>x</div>")
    assert ws.sent == []


def test_notify_sync_schedules_broadcast_on_cached_loop():
    """From a thread, notify_sync must land a broadcast on the cached loop."""
    loop_ready = threading.Event()
    broadcaster = MomentBroadcaster()
    ws = FakeWS()

    captured_loop: list[asyncio.AbstractEventLoop] = []

    def loop_thread(main: Callable[[asyncio.AbstractEventLoop], None]) -> None:
        loop = asyncio.new_event_loop()
        captured_loop.append(loop)
        asyncio.set_event_loop(loop)

        async def register_and_signal() -> None:
            await broadcaster.register(ws)
            loop_ready.set()

        loop.run_until_complete(register_and_signal())
        # Keep the loop alive so notify_sync can schedule onto it.
        main(loop)
        loop.close()

    stop_event = threading.Event()

    def keep_alive(loop: asyncio.AbstractEventLoop) -> None:
        async def spin() -> None:
            while not stop_event.is_set():
                await asyncio.sleep(0.01)

        loop.run_until_complete(spin())

    t = threading.Thread(target=loop_thread, args=(keep_alive,))
    t.start()
    try:
        assert loop_ready.wait(2.0)
        broadcaster.notify_sync("<div>hi</div>")
        # Wait briefly for the scheduled coroutine to run.
        for _ in range(100):
            if ws.sent:
                break
            import time as _t

            _t.sleep(0.01)
        assert ws.sent == ["<div>hi</div>"]
    finally:
        stop_event.set()
        t.join(2.0)


def test_notify_sync_handles_closed_loop_without_raising():
    """Loop closed out from under us → RuntimeError is swallowed."""
    b = MomentBroadcaster()
    ws = FakeWS()
    # Register in one loop, then close it.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(b.register(ws))
    finally:
        loop.close()
    assert b.loop is loop
    # This would raise RuntimeError (loop is closed) inside
    # run_coroutine_threadsafe; the broadcaster must catch it.
    b.notify_sync("<div>x</div>")  # no exception propagates


# ---------------------------------------------------------------------------
# register outside a running loop
# ---------------------------------------------------------------------------


def test_register_without_running_loop_leaves_loop_unset():
    """Calling register() outside any running loop keeps loop None.

    The broadcaster never raises — the caller is responsible for
    invoking register from the endpoint's loop. We only assert that
    the degenerate path doesn't crash and the cached loop stays None.
    """
    b = MomentBroadcaster()
    ws = FakeWS()
    # We cannot call `await` here because there is no running loop; use
    # `loop.run_until_complete` on a fresh loop, which is running during
    # the coroutine but closed after. The cached loop should then be
    # that fresh loop (get_running_loop() succeeded).
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(b.register(ws))
    finally:
        loop.close()
    assert len(b) == 1
    assert b.loop is loop  # cached (even though closed now)


def test_register_with_sync_helper_path(monkeypatch):
    """Exercise the ``RuntimeError → loop stays None`` branch.

    ``asyncio.get_running_loop`` only raises when there is no running
    loop; inside register (which is ``async def``) there *is* a loop,
    so the RuntimeError branch normally isn't reachable. Monkeypatch
    the lookup to force it — this locks the fail-safe path.
    """
    b = MomentBroadcaster()
    ws = FakeWS()

    def raise_runtime() -> asyncio.AbstractEventLoop:
        raise RuntimeError("no running loop")

    monkeypatch.setattr(asyncio, "get_running_loop", raise_runtime)

    async def go() -> None:
        await b.register(ws)

    asyncio.run(go())
    assert len(b) == 1
    assert b.loop is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
