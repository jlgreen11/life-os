"""Async WebSocket broadcaster for Moment state changes.

The :class:`MomentBroadcaster` is the fan-out point that ties the
state-machine-driven backend to the live-updating Now tab. When a
producer creates a new ``SUGGESTED`` Moment or a user accepts /
dismisses one, the route or service hands a pre-rendered HTML fragment
to the broadcaster, which pushes it to every connected client's
WebSocket. HTMX's ``ws`` extension processes the payload for
``hx-swap-oob`` attributes and swaps the right slot in the DOM.

Design notes
------------
- **No global.** The broadcaster is constructed during :class:`LifeOS`
  init (Week 11 wiring) and attached as ``life_os.moment_broadcaster``;
  route handlers reach it via ``request.app.state.life_os`` exactly
  like the repo and feedback store. Tests inject a fresh instance per
  TestClient.
- **Async-native fan-out, sync-safe hook.** The WebSocket endpoint is
  async, so :meth:`broadcast` is a coroutine. FastAPI still runs
  :mod:`api.routes.now`'s action endpoints as sync functions on a
  threadpool (constructor-injection design — §"14-endpoint contract"),
  so sync callers use :meth:`notify_sync`, which thread-safely submits
  a :meth:`broadcast` to the cached main event loop. This avoids
  rewriting every action endpoint as ``async def``.
- **Fail-open.** A send error drops the client and keeps going. The
  endpoint's ``WebSocketDisconnect`` handler is the authoritative
  lifecycle owner; the broadcaster is best-effort.
- **No message framing.** The payload is an HTML string ready for the
  HTMX ``ws`` extension to parse. Any structuring (envelope, metadata)
  belongs in the partial, not here.

References
----------
- CEO plan § "Real-time surface" (WebSocket as the push channel).
- DESIGN.md § "Moment: on arrival — subtle accent bar 3s".
- Engineering plan § "Real-time + full flows" (Week 11).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

log = logging.getLogger(__name__)


class _Sendable(Protocol):
    """Minimal contract the broadcaster needs from a WebSocket-like."""

    async def send_text(self, data: str) -> None: ...


class MomentBroadcaster:
    """Fan out Moment state-change HTML partials to every live client.

    The broadcaster tracks a set of WebSocket clients by identity. A
    single main event loop is captured on the first :meth:`register`
    call so :meth:`notify_sync` can dispatch broadcasts from
    threadpool-hosted sync handlers. Without a cached loop (e.g. if no
    client has ever connected), :meth:`notify_sync` is a silent no-op
    — there is nobody to notify anyway.

    Thread-safety: :meth:`register`/:meth:`unregister`/:meth:`broadcast`
    are intended to run on the main event loop (the WebSocket
    endpoint's loop). :meth:`notify_sync` is the only method callable
    from other threads and uses
    :func:`asyncio.run_coroutine_threadsafe` to hop back on.
    """

    def __init__(self) -> None:
        self._clients: set[_Sendable] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def __len__(self) -> int:
        return len(self._clients)

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The cached main event loop (``None`` until first connect)."""
        return self._loop

    async def register(self, ws: _Sendable) -> None:
        """Add a client and cache the running loop for sync fan-out.

        Safe to call multiple times for the same ``ws`` — the set
        semantics make the add idempotent.
        """
        self._clients.add(ws)
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — we're being called from a sync
                # context. Caller is responsible for calling this on
                # the endpoint's loop; just leave the loop unset.
                self._loop = None

    async def unregister(self, ws: _Sendable) -> None:
        """Remove a client; idempotent — a missing client is a no-op."""
        self._clients.discard(ws)

    async def broadcast(self, html: str) -> int:
        """Send ``html`` to every connected client; return how many saw it.

        Clients that raise on send are dropped from the set. We snapshot
        the iteration target up front so a concurrent ``unregister`` on
        the same loop does not mutate the set underneath us.
        """
        if not self._clients:
            return 0
        sent = 0
        dead: list[_Sendable] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(html)
                sent += 1
            except Exception as exc:
                log.debug("ws send failed; dropping client: %s", exc)
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
        return sent

    def notify_sync(self, html: str) -> None:
        """Schedule a :meth:`broadcast` from a threadpool sync handler.

        This is the bridge that lets :mod:`api.routes.now`'s sync
        action endpoints push state changes without being rewritten
        as ``async def``. No-op when there are no listeners or no loop
        has been cached yet (never-connected process).
        """
        if not self._clients or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(html), self._loop)
        except RuntimeError as exc:
            # The loop was closed out from under us (shutdown race).
            # The WS endpoint will re-cache on next connect.
            log.debug("moment broadcast skipped: %s", exc)


__all__ = ["MomentBroadcaster"]
