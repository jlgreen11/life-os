"""Now-tab + moment-action endpoints.

REST routes locked by engineering plan § "14-endpoint API contract":

- ``GET /``                            → Now-tab page (HTML, full render)
- ``GET /api/now``                     → pending · scheduled · done-today
- ``GET /api/moments/{id}/evidence``   → HTML partial (HTMX reveal)
- ``POST /api/moments/{id}/accept``    → SUGGESTED → ACCEPTED (+ feedback)
- ``POST /api/moments/{id}/dismiss``   → SUGGESTED → DISMISSED (+ feedback)
- ``POST /api/moments/{id}/snooze``    → SUGGESTED → SNOOZED  (+ feedback)
- ``POST /api/moments/{id}/edit``      → update ``proposed_action.params``

Handlers reach service state via ``request.app.state.life_os`` (attached
by :func:`api.app.create_app`). The expected wiring is:

.. code-block:: python

    life_os.moment_repo           # storage.repos.moments.MomentRepository
    life_os.feedback_weight_store # core.moment.feedback_weight.FeedbackWeightStore

Only those two attributes are dereferenced — no implicit global lookup.
If either attribute is missing the handler returns **503** instead of
crashing with an ``AttributeError`` so the rest of the API stays up if
this module is mounted against a half-constructed ``LifeOS``.

Error mapping
-------------
- **404** — ``moment_id`` unknown (or a legacy_task row).
- **409** — state machine refuses the transition.
- **422** — body validation (Pydantic) or missing ``snooze_until``.
- **503** — life_os wiring is incomplete (no ``moment_repo``).

Feedback loop
-------------
Every terminal user decision (accept / dismiss / snooze) is fed into the
:class:`~core.moment.feedback_weight.FeedbackWeightStore` EWMA so the
per-insight-type threshold can self-tune. Edit is *not* a decision and
does not move the weight — it only re-shapes the pending payload.

The outbox dispatch triggered by ``accept`` lands in Week 11 (CEO plan
§ "Outbox pattern spec"); this module records the state transition and
returns the updated Moment. No external side effects fire yet.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, Response

from api.schemas import MomentActionIn, MomentListOut, MomentOut
from core.moment.state import IllegalTransition
from core.moment.types import Moment, MomentState
from web.rendering import render

# Engineering plan § "GET /api/now" locks these limits; keep as module
# constants so tests can import them if we ever want to parameterise.
PENDING_LIMIT = 20
SCHEDULED_LIMIT = 10
DONE_LIMIT = 10

router = APIRouter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _moment_repo(request: Request):
    """Fetch the MomentRepository off ``app.state.life_os`` or 503."""
    life_os = getattr(request.app.state, "life_os", None)
    repo = getattr(life_os, "moment_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="moment_repo is not wired on life_os",
        )
    return repo


def _feedback_store(request: Request):
    """Fetch the FeedbackWeightStore; return ``None`` if missing.

    Unlike the repo, a missing weight store is tolerated — the route
    still transitions the Moment and records it in history. The store
    is logged (via the caller) so operators can notice the gap. This
    mirrors the engine's fail-open stance: a feedback-loop outage must
    not block the user action.
    """
    life_os = getattr(request.app.state, "life_os", None)
    return getattr(life_os, "feedback_weight_store", None)


def _broadcaster(request: Request):
    """Fetch the MomentBroadcaster; return ``None`` if missing.

    The broadcaster is optional wiring — every action endpoint still
    transitions the Moment and returns the usual response even when
    no broadcaster is attached. Its absence just means no live clients
    get pushed the update (they'll pick it up on the next page load).
    """
    life_os = getattr(request.app.state, "life_os", None)
    return getattr(life_os, "moment_broadcaster", None)


def _broadcast_done(broadcaster, moment: Moment) -> None:
    """Push the ``partials/ws_moment_done.html`` OOB swap to all clients.

    No-op when the broadcaster is missing or no clients are connected.
    Called only for terminal states (ACCEPTED / DISMISSED); SNOOZED is
    not a DONE TODAY event — the Moment will resurface at ``snooze_until``
    and re-enter the pending feed at that point.
    """
    if broadcaster is None:
        return
    html = render("partials/ws_moment_done.html", {"moment": moment})
    broadcaster.notify_sync(html)


def _as_moment_out(moment: Moment) -> MomentOut:
    """Convert a :class:`Moment` dataclass into the API schema.

    ``MomentOut`` uses ``extra="forbid"`` so every field must land
    explicitly. The ``state_history`` list is translated row-by-row; all
    other fields map one-to-one.
    """
    return MomentOut(
        id=moment.id,
        created_at=moment.created_at,
        expires_at=moment.expires_at,
        insight=moment.insight,
        evidence=list(moment.evidence),
        evidence_hash=moment.evidence_hash,
        proposed_action={
            "kind": moment.proposed_action.kind,
            "params": dict(moment.proposed_action.params),
        },
        source_insight_type=moment.source_insight_type,
        state=moment.state,
        scheduled_for=moment.scheduled_for,
        context_trigger=(moment.context_trigger.expression if moment.context_trigger else None),
        snooze_until=moment.snooze_until,
        confidence=moment.confidence,
        feedback_weight=moment.feedback_weight,
        state_history=[
            {
                "from_state": entry.from_state,
                "to_state": entry.to_state,
                "ts": entry.ts,
                "annotation": entry.annotation,
            }
            for entry in moment.state_history
        ],
    )


def _load_or_404(repo, moment_id: str) -> Moment:
    """Return the Moment or raise a 404. Legacy-only rows round-trip as 404."""
    moment = repo.get(moment_id)
    if moment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"moment {moment_id!r} not found",
        )
    return moment


def _transition_or_409(repo, moment_id: str, target: MomentState, annotation: str | None) -> Moment:
    """Wrap repo transition; translate ``IllegalTransition`` → 409.

    ``KeyError`` from the repo means the row was deleted between the
    initial GET (which returned 200) and the transition call — treat as
    a 404 so clients see the same shape they would have on first load.
    """
    try:
        return repo.transition(moment_id, target, annotation=annotation)
    except IllegalTransition as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"moment {moment_id!r} not found",
        ) from exc


def _record_feedback(store, insight_type, new_state: MomentState) -> None:
    """Best-effort EWMA update. Silently tolerates a missing store."""
    if store is None:
        return
    try:
        store.update(str(insight_type), new_state)
    except Exception:
        # Fail-open: a feedback outage must never block a user action.
        # Operators see the gap via structured logs at the service layer;
        # we keep the HTTP path quiet.
        return


def _is_htmx(request: Request) -> bool:
    """Return True iff the request was issued by HTMX.

    HTMX sets ``HX-Request: true`` on every fetch it makes; the dual
    JSON / HTML response path keys off this header so non-HTMX API
    clients keep getting JSON ``MomentOut`` payloads while the Now-tab
    UI gets a Moment-card swap partial.
    """
    return request.headers.get("hx-request", "").lower() == "true"


def _next_pending_swap(
    repo,
    *,
    excluding: str,
    moment_id: str,
    previous_state: MomentState,
    new_state: MomentState,
) -> HTMLResponse:
    """Build the HTMX swap response for an action endpoint.

    Renders ``partials/moment_swap.html`` against the highest-confidence
    SUGGESTED Moment that isn't the one we just acted on, falling back
    to an empty placeholder when the queue is drained. Adds an
    ``HX-Trigger`` header carrying the moment_id and previous state so
    the vanilla-JS Undo toast (see ``base.html``) knows what to display
    and what to roll back to once the undo POST endpoint lands.

    The exclusion guard ("``excluding=moment_id``") protects against a
    race window where the just-acted Moment is still indexed as
    SUGGESTED in the read replica — never observed in the synchronous
    test path, but cheap insurance for the eventual replica setup.
    """
    candidates = repo.list_pending(limit=2)
    next_pending: Moment | None = None
    for cand in candidates:
        if cand.id != excluding:
            next_pending = cand
            break

    html = render(
        "partials/moment_swap.html",
        {"next_pending": next_pending},
    )
    headers = {
        "HX-Trigger": json.dumps(
            {
                "lifeos:moment-acted": {
                    "momentId": moment_id,
                    "previousState": previous_state.value,
                    "newState": new_state.value,
                    "action": _action_label(new_state),
                }
            }
        )
    }
    return HTMLResponse(html, headers=headers)


_ACTION_LABEL: dict[MomentState, str] = {
    MomentState.ACCEPTED: "accepted",
    MomentState.DISMISSED: "dismissed",
    MomentState.SNOOZED: "snoozed",
    MomentState.EXPIRED: "snoozed",  # snooze-past-expires path
}


def _action_label(state: MomentState) -> str:
    """Map terminal/post-action MomentState → toast label slug."""
    return _ACTION_LABEL.get(state, state.value)


# ---------------------------------------------------------------------------
# GET /api/now
# ---------------------------------------------------------------------------


@router.get("/api/now", response_model=MomentListOut)
def get_now(request: Request) -> MomentListOut:
    """Return the Now-tab payload.

    Three buckets, each independently limited per engineering plan:

    - ``pending``   — up to 20 SUGGESTED Moments, confidence-ranked.
    - ``scheduled`` — up to 10 Moments whose ``scheduled_for`` is inside
      the default 24h horizon (matches the scheduler's look-ahead).
    - ``done``      — up to 10 Moments whose ``done`` transition
      happened today (UTC), newest first.

    A repo that returns an empty list gives empty lists — never ``None``
    (schema default). Callers never need to null-check.
    """
    repo = _moment_repo(request)
    return MomentListOut(
        pending=[_as_moment_out(m) for m in repo.list_pending(limit=PENDING_LIMIT)],
        scheduled=[_as_moment_out(m) for m in repo.list_scheduled(limit=SCHEDULED_LIMIT)],
        done=[_as_moment_out(m) for m in repo.list_done_today(limit=DONE_LIMIT)],
    )


# ---------------------------------------------------------------------------
# Moment action endpoints
# ---------------------------------------------------------------------------


@router.post("/api/moments/{moment_id}/accept")
def accept_moment(
    moment_id: str,
    request: Request,
    body: MomentActionIn | None = None,
) -> Response:
    """Transition a SUGGESTED moment to ACCEPTED.

    Dual-mode response (engineering plan § "HTMX wiring"):

    - HTMX request (``HX-Request: true``) → ``text/html`` swap partial
      containing the next pending Moment card (or an empty sentinel),
      with ``HX-Trigger: lifeos:moment-acted`` so the Undo toast fires.
    - Anything else (curl, iOS app, JSON test) → ``MomentOut`` JSON.

    The actual outbox dispatch (emailing a draft, creating a calendar
    entry, …) lands in Week 11 — this endpoint only records the state
    change and updates the feedback-weight EWMA. 409 on any non-
    SUGGESTED current state.
    """
    repo = _moment_repo(request)
    existing = _load_or_404(repo, moment_id)
    annotation = body.annotation if body is not None else None
    updated = _transition_or_409(repo, moment_id, MomentState.ACCEPTED, annotation)
    _record_feedback(_feedback_store(request), existing.source_insight_type, MomentState.ACCEPTED)
    _broadcast_done(_broadcaster(request), updated)
    if _is_htmx(request):
        return _next_pending_swap(
            repo,
            excluding=moment_id,
            moment_id=moment_id,
            previous_state=MomentState.SUGGESTED,
            new_state=MomentState.ACCEPTED,
        )
    return _as_moment_out(updated)


@router.post("/api/moments/{moment_id}/dismiss")
def dismiss_moment(
    moment_id: str,
    request: Request,
    body: MomentActionIn | None = None,
) -> Response:
    """Transition a SUGGESTED moment to DISMISSED (terminal).

    Same dual-mode (HTMX HTML / JSON) response shape as
    :func:`accept_moment`. Feeds signal=0.0 into the feedback weight for
    this insight type so the threshold drifts up if the user keeps
    dismissing this producer.
    """
    repo = _moment_repo(request)
    existing = _load_or_404(repo, moment_id)
    annotation = body.annotation if body is not None else None
    updated = _transition_or_409(repo, moment_id, MomentState.DISMISSED, annotation)
    _record_feedback(_feedback_store(request), existing.source_insight_type, MomentState.DISMISSED)
    _broadcast_done(_broadcaster(request), updated)
    if _is_htmx(request):
        return _next_pending_swap(
            repo,
            excluding=moment_id,
            moment_id=moment_id,
            previous_state=MomentState.SUGGESTED,
            new_state=MomentState.DISMISSED,
        )
    return _as_moment_out(updated)


@router.post("/api/moments/{moment_id}/snooze")
def snooze_moment(
    moment_id: str,
    request: Request,
    body: MomentActionIn,
) -> Response:
    """Transition a SUGGESTED moment to SNOOZED and set ``snooze_until``.

    Same dual-mode (HTMX HTML / JSON) response shape as
    :func:`accept_moment`. Requires ``snooze_until`` (unix seconds). A
    value past ``expires_at`` is coerced to EXPIRED by the repo per
    engineering plan § "Snooze semantics"; the endpoint still returns
    200 (with the expired Moment in the JSON path, or the next pending
    swap in the HTMX path) so the client can refresh the row.
    """
    if body.snooze_until is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="snooze_until is required",
        )
    repo = _moment_repo(request)
    existing = _load_or_404(repo, moment_id)
    try:
        updated = repo.snooze(moment_id, body.snooze_until, annotation=body.annotation)
    except IllegalTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"moment {moment_id!r} not found",
        ) from exc
    _record_feedback(_feedback_store(request), existing.source_insight_type, MomentState.SNOOZED)
    if _is_htmx(request):
        return _next_pending_swap(
            repo,
            excluding=moment_id,
            moment_id=moment_id,
            previous_state=MomentState.SUGGESTED,
            new_state=updated.state,
        )
    return _as_moment_out(updated)


# ---------------------------------------------------------------------------
# GET /api/moments/{id}/evidence — HTMX reveal partial
# ---------------------------------------------------------------------------


@router.get("/api/moments/{moment_id}/evidence", response_class=HTMLResponse)
def get_moment_evidence(moment_id: str, request: Request) -> HTMLResponse:
    """Return the evidence-list partial for HTMX reveal.

    Loads the Moment off the repo and renders ``partials/evidence.html``
    against it. The partial returns a list (or an empty-state row) with
    no card chrome — the caller swaps it directly into the panel slot
    inside the parent Moment card.

    Returns 404 if the Moment does not exist (or is a legacy_task row
    filtered out by the repo). The partial is autoescaped, so any
    free-form evidence string is safe to render verbatim.
    """
    repo = _moment_repo(request)
    moment = _load_or_404(repo, moment_id)
    html = render("partials/evidence.html", {"moment": moment})
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# GET / — Now-tab page (server-rendered HTML)
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def now_page(request: Request) -> HTMLResponse:
    """Render the Now tab as a full HTML page.

    Pulls the same three buckets as :func:`get_now` (pending / scheduled /
    done-today) and hands the raw :class:`Moment` dataclasses to the
    Jinja template. The template owns the visual contract (DESIGN.md
    § "Moment card — states"); this handler stays a thin loader so a
    follow-up swap to a different storage layout never has to touch
    template logic.
    """
    repo = _moment_repo(request)
    html = render(
        "now.html",
        {
            "active_tab": "now",
            "pending": list(repo.list_pending(limit=PENDING_LIMIT)),
            "scheduled": list(repo.list_scheduled(limit=SCHEDULED_LIMIT)),
            "done": list(repo.list_done_today(limit=DONE_LIMIT)),
        },
    )
    return HTMLResponse(html)


@router.post("/api/moments/{moment_id}/edit", response_model=MomentOut)
def edit_moment(
    moment_id: str,
    request: Request,
    body: MomentActionIn,
) -> MomentOut:
    """Update ``proposed_action.params`` in place; keep state SUGGESTED.

    Used by the inline draft editor on the Now tab — the user tweaks a
    generated message before accepting. The Moment's state is untouched
    (if it was SUGGESTED it stays SUGGESTED), so callers can chain edit
    then accept without re-fetching. ``action_params`` replaces the
    whole params dict, not a merge — clients must send the complete
    payload.
    """
    if body.action_params is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="action_params is required",
        )
    repo = _moment_repo(request)
    try:
        updated = repo.update_action_params(moment_id, body.action_params)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"moment {moment_id!r} not found",
        ) from exc
    return _as_moment_out(updated)


__all__ = ["DONE_LIMIT", "PENDING_LIMIT", "SCHEDULED_LIMIT", "router"]
