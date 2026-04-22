"""Settings-tab endpoints — connector listing, config, dry-run test.

Three REST routes locked by engineering plan § "14-endpoint API contract":

- ``GET  /api/connectors``               → list every registered connector
- ``PATCH /api/connectors/{id}``         → update enabled / config / secrets
- ``POST /api/connectors/{id}/test``     → dry-run sync (no events written)

All three handlers reach a :class:`ConnectorRepository`-shaped object
off ``request.app.state.life_os.connector_repo``. That repo is the
single integration point between the route layer and whatever owns
connector state on v2 (the repo itself is introduced in the Week 8
settings-repo task and wired on ``LifeOS`` by the orchestrator).

The duck-typed contract is:

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

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import ConnectorConfigIn, ConnectorOut

router = APIRouter()


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


__all__ = ["router"]
