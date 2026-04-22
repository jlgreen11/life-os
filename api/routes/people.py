"""People-tab endpoints — roster and per-contact dossier.

Two REST routes locked by engineering plan § "14-endpoint API contract":

- ``GET /api/people``                → paginated roster + YOU-first
- ``GET /api/people/{contact_id}``   → per-contact dossier

Both handlers reach the :class:`~storage.repos.people.PeopleRepository`
off ``request.app.state.life_os.people_repo``. No other attribute is
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

from fastapi import APIRouter, HTTPException, Query, Request, status

from api.schemas import ContactDossierOut, PeopleListOut
from storage.repos.people import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

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


@router.get("/api/people", response_model=PeopleListOut)
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
) -> PeopleListOut:
    """Return the People-tab roster payload: YOU + two sub-lists + total.

    Empty search / empty DB still returns the full
    :class:`~api.schemas.PeopleListOut` shape — never ``None`` — so the
    UI can render the empty state without null-checking.
    """
    data = _people_repo(request).list_people(q, page, page_size)
    return PeopleListOut(**data)


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


__all__ = ["router"]
