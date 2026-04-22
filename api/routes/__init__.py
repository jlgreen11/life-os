"""v2 API route package.

Each submodule exposes a FastAPI ``APIRouter`` that is mounted by
:func:`api.app.create_app`. Route modules are imported lazily from the
app factory so that schema-only tests (``tests/api/test_schemas.py``)
don't drag FastAPI at import time.

Routers mounted so far
----------------------
- :mod:`api.routes.now` — Week 8 task 1 (Now tab + 4 moment actions).

Subsequent tasks will add ``you``, ``people``, ``settings``, ``health``,
and the ``context`` compat shim under this package.
"""

from __future__ import annotations
