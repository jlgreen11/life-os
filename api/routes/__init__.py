"""v2 API route package.

Each submodule exposes a FastAPI ``APIRouter`` that is mounted by
:func:`api.app.create_app`. Route modules are imported lazily from the
app factory so that schema-only tests (``tests/api/test_schemas.py``)
don't drag FastAPI at import time.

Routers mounted so far
----------------------
- :mod:`api.routes.now`    — Week 8 task 1 (Now tab + 4 moment actions).
- :mod:`api.routes.you`    — Week 8 task 2 (``GET /api/you`` self-portrait).
- :mod:`api.routes.people` — Week 8 task 2 (``GET /api/people`` + dossier).

Subsequent tasks will add ``settings``, ``health``, and the ``context``
compat shim under this package.
"""

from __future__ import annotations
