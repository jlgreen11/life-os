"""People-tab endpoints — roster, per-contact dossier, and HTML pages.

REST + page routes locked by engineering plan § "14-endpoint API contract"
and NEXT_TASKS.md Week 10 ("People tab"):

- ``GET /api/people``                → paginated roster + YOU-first.
                                       Returns JSON ``PeopleListOut`` for
                                       non-HTMX clients. When called by
                                       HTMX (header ``HX-Request: true``)
                                       returns the search-results
                                       partial so the input can swap
                                       ``#people-results`` in place.
- ``GET /api/people/{contact_id}``   → per-contact dossier (JSON).
- ``GET /people``                    → People tab (server-rendered HTML).
- ``GET /people/{contact_id}``       → per-contact dossier (HTML).

All handlers reach the
:class:`~storage.repos.people.PeopleRepository` off
``request.app.state.life_os.people_repo``. No other attribute is
dereferenced; a missing repository yields 503 so the rest of the API
stays up if the module is mounted against a half-constructed LifeOS
(same pattern the Now-tab routes use).

Pagination
----------
Query params on ``GET /api/people``:

- ``q``         — case-insensitive substring filter on contact name
                  / id. Empty / missing means "no filter".
- ``page``      — 1-indexed page number (default 1, min 1).
- ``page_size`` — rows per page (default
                  :data:`storage.repos.people.DEFAULT_PAGE_SIZE`, max
                  :data:`storage.repos.people.MAX_PAGE_SIZE`).

The payload's two sub-lists (``needs_attention`` / ``active_this_week``)
are filtered *within the current page* — a contact that falls off the
end of the page is not carried forward to either list. ``total``
reflects the full post-filter roster so clients can drive a pager.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse

from api.schemas import ContactDossierOut, PeopleListOut
from storage.repos.people import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
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


def _is_htmx(request: Request) -> bool:
    """Return True iff the request was issued by HTMX.

    Mirrors :func:`api.routes.now._is_htmx` so the dual JSON / HTML
    response path keys off the same header on every tab.
    """
    return request.headers.get("hx-request", "").lower() == "true"


@router.get("/api/people", response_model=None)
def list_people(
    request: Request,
    q: str | None = Query(default=None, description="Substring filter"),
    page: int = Query(default=1, ge=1, description="1-indexed page"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Rows per page",
    ),
) -> Response:
    """Return the People-tab roster payload.

    Dual response shape (engineering plan § "HTMX wiring"):

    - HTMX request → ``text/html`` partial containing the YOU pinned
      row plus the two sub-list sections, suitable for an
      ``hx-swap="innerHTML"`` target of ``#people-results``.
    - Anything else → JSON ``PeopleListOut``. Empty search / empty DB
      still returns the full schema shape — never ``None`` — so the UI
      can render the empty state without null-checking.
    """
    data = _people_repo(request).list_people(q, page, page_size)
    if _is_htmx(request):
        html = render("partials/people_results.html", {"people": data})
        return HTMLResponse(html)
    # Validate against the schema before returning so a repo drift fails
    # loudly instead of silently shipping a bad payload.
    return Response(
        content=PeopleListOut(**data).model_dump_json(),
        media_type="application/json",
    )


@router.get("/api/people/{contact_id}", response_model=ContactDossierOut)
def get_dossier(contact_id: str, request: Request) -> ContactDossierOut:
    """Return the full dossier for a single contact.

    404 when the contact is unknown to both the entities table and
    every per-contact signal profile namespace. 503 when the repository
    is not wired.
    """
    data = _people_repo(request).get_dossier(contact_id)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"contact {contact_id!r} not found",
        )
    return ContactDossierOut(**data)


@router.get("/people", response_class=HTMLResponse)
def people_page(
    request: Request,
    q: str | None = Query(default=None, description="Substring filter"),
    page: int = Query(default=1, ge=1, description="1-indexed page"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Rows per page",
    ),
) -> HTMLResponse:
    """Render the People tab as a full HTML page.

    Loads the roster off the repository (same call the JSON endpoint
    uses) and hands it to ``people.html`` which embeds the search input
    and the same partial the HTMX swap returns. Missing ``people_repo``
    wiring yields 503, matching :func:`list_people` so both the page
    and the API degrade identically when the orchestrator is
    half-constructed.
    """
    data = _people_repo(request).list_people(q, page, page_size)
    html = render(
        "people.html",
        {
            "active_tab": "people",
            "people": data,
        },
    )
    return HTMLResponse(html)


@router.get("/people/{contact_id}", response_class=HTMLResponse)
def contact_page(contact_id: str, request: Request) -> HTMLResponse:
    """Render the per-contact dossier as a full HTML page.

    Returns 404 for unknown contacts, matching :func:`get_dossier` so
    the JSON and HTML routes share an error contract. 503 when the
    repository is not wired.
    """
    data = _people_repo(request).get_dossier(contact_id)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"contact {contact_id!r} not found",
        )
    html = render(
        "contact_dossier.html",
        {
            "active_tab": "people",
            "contact": data,
        },
    )
    return HTMLResponse(html)


__all__ = ["router"]
