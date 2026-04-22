"""FastAPI application factory for the v2 API.

Single entry point for constructing the v2 API surface. Routes are
registered in Weeks 8-11 under ``api/routes/``; this module holds only
the factory and its wiring. The v1 ``web/`` package stays untouched
until cutover (Week 12) so both apps can coexist during dry-run.

Design notes
------------
- **Application-factory pattern.** Each call to :func:`create_app`
  returns a fresh :class:`fastapi.FastAPI` instance. Tests can build
  their own app against a ``LifeOS`` double without module-level
  side effects (same pattern as v1's ``web.app.create_web_app``).

- **CORS locked to Tailscale-only localhost** in Phase 1 (engineering
  plan § "Security posture"). The factory reads an explicit allow-list
  from ``life_os.config["cors"]["allowed_origins"]`` and falls back to
  a localhost-only default; the wildcard origin is never accepted.

- **No service imports at import time.** Route modules will land
  under ``api/routes/*`` and will be imported lazily by the registrar
  added in the Now/You/People/Settings tasks. Keeping this skeleton
  free of service imports means :mod:`api.app` is round-trippable by
  a type checker without pulling in the whole stack.

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md`` § "API surface".
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)


def _resolve_allowed_origins(life_os: Any) -> list[str]:
    """Extract the CORS allow-list from the LifeOS config.

    Invalid, missing, or empty lists fall back to the secure localhost-
    only default. The wildcard origin ``"*"`` is never allowed — the
    fallback path always returns the concrete localhost tuple. This
    mirrors v1's behaviour (see ``web/app.py::create_web_app``) so the
    cutover does not loosen the security posture.
    """
    config = getattr(life_os, "config", {}) or {}
    cors_config = config.get("cors", {}) if isinstance(config, dict) else {}
    raw = cors_config.get("allowed_origins") if isinstance(cors_config, dict) else None

    if not isinstance(raw, list):
        return list(_DEFAULT_ALLOWED_ORIGINS)

    cleaned = [origin.strip() for origin in raw if isinstance(origin, str) and origin.strip() and origin.strip() != "*"]
    if not cleaned:
        return list(_DEFAULT_ALLOWED_ORIGINS)
    return cleaned


def create_app(life_os: Any | None = None) -> FastAPI:
    """Build a configured :class:`fastapi.FastAPI` for the v2 API.

    Parameters
    ----------
    life_os:
        Optional LifeOS orchestrator. When provided, services and
        configuration are reachable from route handlers via
        ``request.app.state.life_os``. In tests, a lightweight object
        (or ``None``) is acceptable — the skeleton itself does not
        dereference service attributes.

    Returns
    -------
    FastAPI
        A fresh application with CORS middleware installed and the
        ``life_os`` instance attached to ``app.state``. No routes are
        registered by this skeleton; route wiring lands in subsequent
        Week 8 tasks.
    """

    app = FastAPI(
        title="Life OS v2",
        description="Evidence-backed, state-machine-governed Moments API.",
        version="2.0.0-dev",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_allowed_origins(life_os),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.life_os = life_os

    # Route registration is deferred to keep ``api.app`` importable from
    # schema-only contexts. Each router is imported here, not at module
    # top-level, so test suites that only build the factory don't drag
    # route modules they're not exercising.
    from api.routes.context import router as context_router
    from api.routes.health import router as health_router
    from api.routes.now import router as now_router
    from api.routes.people import router as people_router
    from api.routes.settings import router as settings_router
    from api.routes.you import router as you_router

    app.include_router(now_router)
    app.include_router(you_router)
    app.include_router(people_router)
    app.include_router(settings_router)
    app.include_router(health_router)
    app.include_router(context_router)

    return app


__all__ = ["create_app"]
