"""Life OS v2 — API package.

Fresh-module rewrite of v1's ``web/routes.py``. This package holds the
FastAPI application factory (:func:`api.app.create_app`) and the
Pydantic request/response schemas (:mod:`api.schemas`). Route modules
land under ``api/routes/`` in Weeks 8-11 and are registered by
:func:`api.app.create_app`.

The schemas are locked by the engineering plan (docs/plans/
2026-04-21-v2-rewrite-plan.md) against the 14 REST endpoints plus the
iOS compat shim. Wire shape is intentionally thin: the skeleton does
not import any repo or service layer so it can be validated by
type-checkers and Pydantic alone — no runtime DB is needed to
round-trip a payload.
"""

# Lazy re-exports: ``api.schemas`` and ``api.app`` are imported on demand
# so that unit tests against just the schemas do not need FastAPI/starlette
# resolvable at import time (the AI engine module follows the same pattern).
# Callers who need the factory say ``from api.app import create_app`` or
# ``from api import create_app`` and the __getattr__ hook below loads it.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ActionOut",
    "ConnectorConfigIn",
    "ConnectorOut",
    "ContactDossierOut",
    "ContactSummaryOut",
    "DriftingContactOut",
    "HealthOut",
    "MetricsOut",
    "MomentActionIn",
    "MomentListOut",
    "MomentOut",
    "PeopleListOut",
    "PersonaStyleOut",
    "RoutineOut",
    "StateHistoryOut",
    "YouOut",
    "create_app",
]

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from api.app import create_app
    from api.schemas import (
        ActionOut,
        ConnectorConfigIn,
        ConnectorOut,
        ContactDossierOut,
        ContactSummaryOut,
        DriftingContactOut,
        HealthOut,
        MetricsOut,
        MomentActionIn,
        MomentListOut,
        MomentOut,
        PeopleListOut,
        PersonaStyleOut,
        RoutineOut,
        StateHistoryOut,
        YouOut,
    )


def __getattr__(name: str) -> Any:
    """Lazy attribute access: import submodules only when first needed."""
    if name == "create_app":
        from api.app import create_app

        return create_app
    if name in __all__:
        from api import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module 'api' has no attribute {name!r}")
