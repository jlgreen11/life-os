"""You-tab endpoint — self-portrait payload.

One REST route locked by engineering plan § "14-endpoint API contract":

- ``GET /api/you`` → self-portrait (observed_months, interactions_count,
  confidence_pct, when_at_best, how_you_write, your_routines, drifting).

The handler reaches the :class:`~storage.repos.people.PeopleRepository`
via ``request.app.state.life_os.people_repo``. No other attribute is
dereferenced — if the repository is missing the route returns 503 (same
pattern the Now-tab routes use) rather than crashing with an
``AttributeError``.

Empty-state contract
--------------------
Every list section on :class:`~api.schemas.YouOut` carries a schema-level
default of ``[]``; the repository likewise returns ``[]`` when no
producer has written a profile yet. Clients therefore never see
``None`` for a section, and the JSON shape is identical between a fresh
install and a fully-populated one.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from api.schemas import YouOut
from web.rendering import render

router = APIRouter()


def _people_repo(request: Request):
    """Fetch the PeopleRepository off ``app.state.life_os`` or 503."""
    life_os = getattr(request.app.state, "life_os", None)
    repo = getattr(life_os, "people_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="people_repo is not wired on life_os",
        )
    return repo


@router.get("/api/you", response_model=YouOut)
def get_you(request: Request) -> YouOut:
    """Return the self-portrait for the You tab.

    The repo guarantees every field is present; we pass the raw dict
    straight into :class:`YouOut` so Pydantic's ``extra='forbid'`` catches
    any drift between the SQL layer and the wire schema at first request.
    """
    return YouOut(**_people_repo(request).get_you())


@router.get("/you", response_class=HTMLResponse)
def you_page(request: Request) -> HTMLResponse:
    """Render the You tab as a full HTML page.

    Loads the self-portrait dict straight off the repository (same call
    the JSON endpoint uses) and hands it to the Jinja template. Missing
    ``people_repo`` wiring yields 503, matching :func:`get_you` so both
    the page and the API degrade identically when the orchestrator is
    half-constructed.
    """
    you_data = _people_repo(request).get_you()
    html = render(
        "you.html",
        {
            "active_tab": "you",
            "you": you_data,
        },
    )
    return HTMLResponse(html)


__all__ = ["router"]
