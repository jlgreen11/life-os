"""Settings-tab endpoints — connector listing, config, dry-run test, page.

REST + page routes locked by engineering plan § "14-endpoint API contract"
and NEXT_TASKS.md Week 10 ("Settings tab"):

- ``GET  /api/connectors``               → list every registered connector
- ``PATCH /api/connectors/{id}``         → update enabled / config / secrets
- ``POST /api/connectors/{id}/test``     → dry-run sync (no events written)
- ``GET  /settings``                     → Settings tab (server-rendered HTML)
- ``GET  /settings/connectors/{id}/edit``→ edit-form partial for the detail pane
- ``POST /settings/connectors/{id}/test``→ dry-run sync + HTML result partial
- ``GET  /settings/detail-empty``        → empty-state for the detail pane
                                            (used by the edit form's Cancel)

The three JSON handlers reach a :class:`ConnectorRepository`-shaped
object off ``request.app.state.life_os.connector_repo``. The HTML
handlers extend the duck-typed contract with two optional methods that
the storage-backed repo is expected to implement:

- ``edit_view(connector_id) -> dict | None``
    Returns a dict keyed ``{id, kind, enabled, status, config,
    secret_keys}`` where ``config`` is the non-secret portion and
    ``secret_keys`` lists the *names* of fields whose values live behind
    the Fernet boundary. Values are NEVER returned — the edit form
    renders empty placeholders and a blank submission means "keep
    existing" (see the partial's header).
- ``list() / update() / test()`` — unchanged.

When ``edit_view`` is absent, the HTML edit route synthesises a minimal
view from ``list()`` so the page still renders against a stub repo.

Duck-typed contract (unchanged from the skeleton)
-------------------------------------------------
- ``list() -> list[dict]``
    One row per connector known to the registry; keys match
    :class:`api.schemas.ConnectorOut` (``id``, ``kind``, ``enabled``,
    ``status``, ``last_sync_at``, ``last_error``).
- ``update(connector_id, *, enabled, config, secrets) -> dict | None``
    Applies the patch; returns the updated row in ``ConnectorOut`` shape,
    or ``None`` when the id is unknown. Any Fernet-encrypted secret stays
    behind the repo boundary and NEVER appears in the returned dict.
- ``test(connector_id) -> dict | None``
    Performs a dry-run sync (no persisted events / outbox writes); returns
    a free-form status dict (``{"ok": bool, "message": str, ...}``) or
    ``None`` when the id is unknown.

Preferences on the page render
------------------------------
The ``GET /settings`` handler reads four preference keys off
``life_os.db`` (the same ``sqlite3.Connection`` the context shim uses):

    quiet_hours_start  (default "22:00")
    quiet_hours_end    (default "07:00")
    autonomy_level     (default 0.5)
    proactivity        (default 0.5)

Missing ``life_os.db`` is not fatal — the page renders with the schema-
level defaults so a fresh install still loads. Persisting the fields goes
through the existing ``POST /api/preferences`` shim (iOS-compat route),
which both the form and tests exercise unchanged.

Error mapping
-------------
- **404** — unknown ``connector_id``.
- **422** — request body validation (Pydantic, ``extra='forbid'``).
- **503** — ``connector_repo`` is not wired on life_os.

Security
--------
Fernet-encrypted credentials are **never** returned by this module.
:class:`api.schemas.ConnectorOut` deliberately omits every secret field;
the repo layer is responsible for masking. The PATCH handler trusts its
transport (Tailscale-only in Phase 1, engineering plan § "Security
posture") to carry plaintext secrets, re-encrypts them server-side, and
returns only status-level fields.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from api.schemas import ConnectorConfigIn, ConnectorOut
from web.rendering import render

router = APIRouter()


_DETAIL_EMPTY_HTML = (
    '<div data-empty="detail-pane">'
    '<p class="text-[length:var(--t-15)] text-[color:var(--text-secondary)] m-0">'
    "Select a connector to edit."
    "</p>"
    '<p class="text-[length:var(--t-13)] text-[color:var(--text-tertiary)] mt-[var(--s-1)] m-0">'
    "Changes save on submit. Credentials are encrypted at rest."
    "</p>"
    "</div>"
)


_PREFERENCE_DEFAULTS: dict[str, Any] = {
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "07:00",
    "autonomy_level": 0.5,
    "proactivity": 0.5,
}


def _connector_repo(request: Request):
    """Fetch the ConnectorRepository off ``app.state.life_os`` or 503.

    Same fail-soft pattern the Now/People routes use — a missing repo
    means the module was mounted against a half-constructed LifeOS, and
    the right behaviour is to keep the rest of the API up while this
    slice returns an explicit 503.
    """
    life_os = getattr(request.app.state, "life_os", None)
    repo = getattr(life_os, "connector_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="connector_repo is not wired on life_os",
        )
    return repo


def _load_preferences(request: Request) -> dict[str, Any]:
    """Read the four page-render preferences off ``life_os.db``.

    Silently falls back to schema-level defaults when:
      - ``life_os`` is missing (tests),
      - ``life_os.db`` is None (partial wiring),
      - the ``preferences`` table is absent (fresh install),
      - a row exists but its JSON is malformed (never panic on load).

    Numeric values are coerced to floats; strings pass through untouched.
    Defaults live in ``_PREFERENCE_DEFAULTS`` so tests can import and
    assert on the same constant.
    """
    values: dict[str, Any] = dict(_PREFERENCE_DEFAULTS)
    life_os = getattr(request.app.state, "life_os", None)
    conn: sqlite3.Connection | None = getattr(life_os, "db", None)
    if conn is None:
        return values
    try:
        rows = conn.execute(
            "SELECT key, value FROM preferences WHERE key IN (?, ?, ?, ?)",
            tuple(_PREFERENCE_DEFAULTS.keys()),
        ).fetchall()
    except sqlite3.Error:
        return values
    for key, raw in rows:
        if raw is None:
            continue
        # Every value is JSON-encoded on the write path (see context shim);
        # fall back to the raw string if decoding fails so a hand-edited
        # row doesn't blank the field.
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            decoded = raw
        if key in ("autonomy_level", "proactivity"):
            try:
                values[key] = float(decoded)
            except (TypeError, ValueError):
                values[key] = _PREFERENCE_DEFAULTS[key]
        else:
            values[key] = decoded if isinstance(decoded, str) else str(decoded)
    return values


@router.get("/api/connectors", response_model=list[ConnectorOut])
def list_connectors(request: Request) -> list[ConnectorOut]:
    """Return every registered connector plus its live state.

    The repo is expected to join the static connector registry
    (``connectors/registry.py``) with the ``connector_state`` table and
    the preferences-backed ``enabled`` flag. An install with zero
    configured connectors still round-trips the registry — the UI never
    has to distinguish "empty response" from "repo returned None".
    """
    repo = _connector_repo(request)
    return [ConnectorOut(**row) for row in repo.list()]


@router.patch("/api/connectors/{connector_id}", response_model=ConnectorOut)
def patch_connector(
    connector_id: str,
    body: ConnectorConfigIn,
    request: Request,
) -> ConnectorOut:
    """Update a connector's enabled flag, config, and/or encrypted secrets.

    All three fields on :class:`ConnectorConfigIn` are optional — a PATCH
    can toggle only ``enabled`` without re-sending creds. Fields that are
    present are fully replaced (no merge); the repo re-encrypts any
    ``secrets`` entries before writing them to preferences.

    Returns the post-update row in :class:`ConnectorOut` shape. The
    response never carries the raw secrets that the request carried —
    only status-level fields. 404 when the id is unknown to the repo.
    """
    repo = _connector_repo(request)
    updated = repo.update(
        connector_id,
        enabled=body.enabled,
        config=body.config,
        secrets=body.secrets,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"connector {connector_id!r} not found",
        )
    return ConnectorOut(**updated)


@router.post("/api/connectors/{connector_id}/test")
def test_connector(connector_id: str, request: Request) -> dict[str, Any]:
    """Dry-run a sync against the connector and return a status dict.

    "Dry-run" means: authenticate, probe the external service, but do NOT
    publish events to NATS, do NOT write to the outbox, and do NOT mutate
    ``connector_state.cursor``. The repo is responsible for enforcing
    that invariant; the route simply forwards the request.

    Shape is intentionally free-form (not a Pydantic model) so each
    connector can surface connector-specific diagnostics (IMAP folder
    counts, Plaid institution name, …) without churning a shared schema.
    A sensible default is ``{"ok": bool, "message": str, "details": dict}``.

    404 when the id is unknown. 503 when the repo is not wired.
    """
    repo = _connector_repo(request)
    result = repo.test(connector_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"connector {connector_id!r} not found",
        )
    return result


# ---------------------------------------------------------------------------
# HTML page + detail-pane partials
# ---------------------------------------------------------------------------


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    """Render the Settings tab as a full HTML page.

    Loads the connector list off the repo (same call the JSON endpoint
    uses) and the four page-render preferences off ``life_os.db``, then
    hands them to :file:`web/templates/settings.html`. Missing
    ``connector_repo`` yields 503, matching :func:`list_connectors` so
    both the page and the API degrade identically when the orchestrator
    is half-constructed.
    """
    repo = _connector_repo(request)
    connectors = list(repo.list())
    preferences = _load_preferences(request)
    html = render(
        "settings.html",
        {
            "active_tab": "settings",
            "connectors": connectors,
            "preferences": preferences,
        },
    )
    return HTMLResponse(html)


@router.get("/settings/detail-empty", response_class=HTMLResponse)
def settings_detail_empty() -> HTMLResponse:
    """Render the initial / cancel state for the detail pane.

    Returned on first load (inlined in the page template) and by the
    edit form's Cancel button via HTMX. Kept as a fixed HTML snippet
    here (not a template file) because it's three lines — a partial
    file would be more ceremony than the content earns.
    """
    return HTMLResponse(_DETAIL_EMPTY_HTML)


@router.get("/settings/connectors/{connector_id}/edit", response_class=HTMLResponse)
def settings_edit_connector(connector_id: str, request: Request) -> HTMLResponse:
    """Return the HTML edit form for the detail pane.

    Prefers ``repo.edit_view(id)`` (which can surface per-connector
    config / secret_keys). When absent, synthesises a minimal view from
    ``list()`` so the page still works against the test stub. 404 when
    the id is unknown.
    """
    repo = _connector_repo(request)
    view: dict[str, Any] | None
    edit_view_fn = getattr(repo, "edit_view", None)
    if callable(edit_view_fn):
        view = edit_view_fn(connector_id)
    else:
        view = next((r for r in repo.list() if r.get("id") == connector_id), None)
        if view is not None:
            # Minimal synthesis: stubs don't surface config / secret_keys.
            view = dict(view)
            view.setdefault("config", {})
            view.setdefault("secret_keys", [])
    if view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"connector {connector_id!r} not found",
        )
    html = render("partials/connector_edit_form.html", {"connector": view})
    return HTMLResponse(html)


@router.post("/settings/connectors/{connector_id}/test", response_class=HTMLResponse)
def settings_test_connector(connector_id: str, request: Request) -> HTMLResponse:
    """Dry-run the connector and return the result as an HTML partial.

    Wraps :func:`test_connector` so the test button can target a simple
    ``[data-slot="test-result"]`` div without parsing JSON client-side.
    The partial renders ``✓ message`` on success and ``✗ message`` on
    failure. 404 propagates unchanged so HTMX surfaces the error.
    """
    repo = _connector_repo(request)
    result = repo.test(connector_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"connector {connector_id!r} not found",
        )
    html = render("partials/connector_test_result.html", {"result": result})
    return HTMLResponse(html)


__all__ = ["router"]
