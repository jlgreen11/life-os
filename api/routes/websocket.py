"""WebSocket endpoint for live Moment push updates.

Mounts a single ``/ws`` WebSocket route that HTMX's ``ws`` extension
connects to from the base template. Once accepted, the socket is
passed to :class:`~core.moment.broadcaster.MomentBroadcaster` so any
subsequent state change — a new SUGGESTED Moment from a producer, or
an ACCEPTED/DISMISSED transition from the user — pushes an HTML
partial out to the client.

Wire contract
-------------
Clients never send payloads upstream on this socket; the server only
broadcasts. :func:`WebSocket.receive_text` is used purely as a
blocking wait for a disconnect, which surfaces as
:class:`fastapi.WebSocketDisconnect`. The endpoint's ``finally`` block
is the single place that removes a client from the broadcaster, so
a dropped connection always cleans up regardless of how it terminated.

503 path
--------
When ``life_os.moment_broadcaster`` is not wired (half-constructed
app — same failure mode the Now routes surface), the endpoint closes
the socket with code 1011 (server error) and an explanatory reason.
Accepting first and then closing is deliberate: an unaccepted close
looks like a handshake refusal and makes browser-side reconnection
loops hammer the server. A clean close-after-accept tells the client
"we heard you, but the server isn't ready" and the HTMX ws extension
backs off per its configured policy.

References
----------
- CEO plan § "Real-time surface".
- DESIGN.md § "Moment: on arrival — subtle accent bar 3s".
- Engineering plan § "Real-time + full flows" (Week 11).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

router = APIRouter()

# Matches the WebSocket policy-violation / internal-error close codes
# (RFC 6455 § 7.4.1). 1011 signals "server error" which is what a
# missing broadcaster represents from the client's perspective.
_CLOSE_NO_BROADCASTER = 1011


def _broadcaster(websocket: WebSocket):
    """Return the broadcaster off ``app.state.life_os`` or ``None``.

    Keeping the lookup side-effect-free means the endpoint can decide
    to close gracefully (accept → close) instead of refusing the
    handshake outright, which browsers treat as a hard failure and
    retry aggressively.
    """
    life_os = getattr(websocket.app.state, "life_os", None)
    return getattr(life_os, "moment_broadcaster", None)


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Accept a connection, register with the broadcaster, hold open.

    Flow:

    1. Resolve the broadcaster; if missing, accept + close with 1011.
    2. Accept the handshake and register the socket so subsequent
       broadcasts include it in the fan-out set.
    3. Block on :meth:`WebSocket.receive_text` until the client hangs
       up; we do not consume any client messages.
    4. Always unregister on exit so the broadcaster's set stays clean.
    """
    broadcaster = _broadcaster(websocket)
    if broadcaster is None:
        await websocket.accept()
        await websocket.close(code=_CLOSE_NO_BROADCASTER, reason="broadcaster not wired")
        return

    await websocket.accept()
    await broadcaster.register(websocket)
    try:
        while True:
            # Clients don't push state upstream — we just park until
            # the socket drops. ``receive_text`` raises
            # ``WebSocketDisconnect`` on a clean close; other errors
            # fall through to the ``finally`` below.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.unregister(websocket)


__all__ = ["router"]
