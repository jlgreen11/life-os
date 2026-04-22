"""Fixtures for the Now-tab end-to-end browser tests.

Spins up the v2 :func:`api.app.create_app` against a real
:class:`~storage.repos.moments.MomentRepository` (in-memory SQLite),
a real :class:`~core.moment.feedback_weight.FeedbackWeightStore`, and
a real :class:`~core.moment.broadcaster.MomentBroadcaster`, then runs
the FastAPI app inside a background uvicorn server so a real browser
(Playwright preferred, Selenium fallback) can drive it.

Why a real server (not :class:`fastapi.testclient.TestClient`)
-------------------------------------------------------------
TestClient handles HTTP and WebSockets in process, but it does not
execute HTMX, JS event handlers, or WebSocket OOB swaps. The five
critical-path flows in NEXT_TASKS.md ("accept moves card out", "WS
push removes accepted card via OOB", …) are JavaScript-driven, so we
need a browser engine and an HTTP listener it can reach.

Skip-with-warning behavior
--------------------------
``test_now_tab_e2e.py`` carries a module-level ``pytestmark`` that
skips the entire module when neither Playwright nor Selenium is
importable. This module emits a single ``UserWarning`` in the same
case so CI / the orchestrator can surface "browser-driven E2E suite
was skipped because no driver was installed" without parsing pytest
short-summary output. The warning fires at fixture-collection time,
not import time, so a clean ``pytest --collect-only`` against a
host without browser deps still succeeds quietly.

Per-test isolation
------------------
Each test gets a fresh SQLite (``:memory:``), a fresh repo, a fresh
broadcaster, and a fresh uvicorn server on an ephemeral port. The
server is shut down in fixture teardown so a flaky test cannot leak
a bound port to the next one. Browser context is also recycled per
test (Playwright: ``browser.new_context()``; Selenium: a fresh
``WebDriver`` per test via the chromedriver session) so cookies,
WebSocket connections, and the like never bleed across tests.
"""

from __future__ import annotations

import socket
import sqlite3
import threading
import time
import warnings
from collections.abc import Iterator
from contextlib import closing
from typing import Any

import pytest
import uvicorn

from api.app import create_app
from core.moment.broadcaster import MomentBroadcaster
from core.moment.feedback_weight import FeedbackWeightStore
from storage import schema
from storage.repos.moments import MomentRepository

# Fixed reference epoch — 2026-04-22T12:00:00Z. Mirrors the route tests
# so any time-sensitive list (e.g. list_done_today) is deterministic.
REF_NOW = 1_777_204_800


# ---------------------------------------------------------------------------
# Driver discovery (Playwright preferred, Selenium fallback)
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

    DRIVER: str | None = "playwright"
except Exception:  # pragma: no cover — exercised via skipif on host machines without playwright.
    sync_playwright = None  # type: ignore[assignment]
    try:
        import selenium.webdriver as _selenium_webdriver  # type: ignore[import-not-found]  # noqa: F401

        DRIVER = "selenium"
    except Exception:
        DRIVER = None
        warnings.warn(
            "tests/e2e: neither playwright nor selenium is installed; "
            "Now-tab browser-driven E2E suite will be skipped. "
            "Install playwright with: pip install playwright && playwright install chromium.",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# In-process FastAPI server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return a free localhost TCP port for the test server.

    We bind, read the chosen port, and close — the OS may hand the
    same port to another process before uvicorn binds it, but the
    window is small enough that a single retry inside the test
    runner is unnecessary in practice.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _DummyLifeOS:
    """Minimal life_os double the v2 routes dereference.

    Mirrors ``tests/api/test_routes_now.py::DummyLifeOS`` but adds the
    ``moment_broadcaster`` attribute the WS route + the action endpoints
    use to fan out state-change partials.
    """

    def __init__(
        self,
        *,
        moment_repo: Any,
        feedback_weight_store: Any,
        moment_broadcaster: Any,
    ) -> None:
        self.config: dict[str, Any] = {}
        self.moment_repo = moment_repo
        self.feedback_weight_store = feedback_weight_store
        self.moment_broadcaster = moment_broadcaster
        self.outbox_repo = None  # outbox dispatch is out of scope for these flows


class _BackgroundUvicorn:
    """Run a uvicorn server in a daemon thread for the duration of one test.

    Why not :class:`fastapi.testclient.TestClient`? See the module
    docstring — TestClient is in-process and a real browser cannot
    reach it. We instantiate :class:`uvicorn.Server` directly so the
    ``should_exit`` flag gives us a clean teardown without ``os._exit``
    games, and so we can poll ``server.started`` instead of guessing
    a sleep time.
    """

    def __init__(self, app: Any, port: int) -> None:
        self.port = port
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",  # no startup/shutdown hooks in the v2 skeleton.
        )
        self.server = uvicorn.Server(config)
        # Suppress uvicorn's signal-handler install — only the main thread can.
        self.server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout: float = 5.0) -> None:
        self.thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError(f"uvicorn did not start within {timeout}s on port {self.port}")

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Fresh in-memory SQLite with the full v2 schema and FKs on."""
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def clock() -> Any:
    """Monotonically advanceable clock — same shape as the route tests."""

    class _Clock:
        def __init__(self, t: float = REF_NOW) -> None:
            self.t = t

        def __call__(self) -> float:
            return self.t

    return _Clock()


@pytest.fixture
def repo(conn: sqlite3.Connection, clock: Any) -> MomentRepository:
    return MomentRepository(conn, now_fn=clock)


@pytest.fixture
def feedback(conn: sqlite3.Connection, clock: Any) -> FeedbackWeightStore:
    return FeedbackWeightStore(conn, now_fn=clock)


@pytest.fixture
def broadcaster() -> MomentBroadcaster:
    return MomentBroadcaster()


@pytest.fixture
def server(
    repo: MomentRepository,
    feedback: FeedbackWeightStore,
    broadcaster: MomentBroadcaster,
) -> Iterator[_BackgroundUvicorn]:
    """Spin up the v2 FastAPI app on an ephemeral port; tear down on exit."""
    life_os = _DummyLifeOS(
        moment_repo=repo,
        feedback_weight_store=feedback,
        moment_broadcaster=broadcaster,
    )
    app = create_app(life_os)
    bg = _BackgroundUvicorn(app, _free_port())
    bg.start()
    try:
        yield bg
    finally:
        bg.stop()
