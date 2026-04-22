"""Tests for :mod:`api.app` — the v2 FastAPI application factory.

These are the skeleton smoke tests that land before any route module.
They cover:

- The factory returns a fresh :class:`fastapi.FastAPI` instance each
  call (application-factory invariant).
- ``life_os`` is attached to ``app.state`` so downstream routes can
  reach services without module-level globals.
- The CORS middleware default is localhost-only when no config is
  passed, and invalid / wildcard values fall back to that default
  (engineering plan § "Security posture": no ``*`` origins).
- A well-formed allow-list is honoured as-is.

No route tests live here — the skeleton registers no routes. Later
Week 8 tasks extend this module with ``TestClient`` coverage for
``/api/now``, ``/api/you``, etc.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.app import create_app


class _DummyLifeOS:
    def __init__(self, config):
        self.config = config


def _cors_middleware(app: FastAPI):
    for mw in app.user_middleware:
        if mw.cls.__name__ == "CORSMiddleware":
            return mw
    raise AssertionError("CORSMiddleware not registered")


def test_create_app_returns_fresh_fastapi_instance():
    a1 = create_app()
    a2 = create_app()

    assert isinstance(a1, FastAPI)
    assert isinstance(a2, FastAPI)
    assert a1 is not a2
    assert a1.title == "Life OS v2"
    assert a1.version == "2.0.0-dev"


def test_create_app_attaches_life_os_to_state():
    life_os = _DummyLifeOS(config={"cors": {"allowed_origins": ["http://localhost:8080"]}})

    app = create_app(life_os)

    assert app.state.life_os is life_os


def test_create_app_defaults_cors_to_localhost_when_no_config():
    app = create_app(None)

    mw = _cors_middleware(app)
    origins = mw.kwargs["allow_origins"]
    assert origins == ["http://localhost:8080", "http://127.0.0.1:8080"]


def test_create_app_defaults_when_origins_malformed():
    life_os = _DummyLifeOS(config={"cors": {"allowed_origins": "https://bad"}})

    app = create_app(life_os)

    mw = _cors_middleware(app)
    assert mw.kwargs["allow_origins"] == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


def test_create_app_strips_wildcard_origin():
    life_os = _DummyLifeOS(config={"cors": {"allowed_origins": ["*"]}})

    app = create_app(life_os)

    mw = _cors_middleware(app)
    assert "*" not in mw.kwargs["allow_origins"]
    assert mw.kwargs["allow_origins"] == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


def test_create_app_honours_valid_allow_list():
    origins = ["http://localhost:3000", "http://127.0.0.1:8080"]
    life_os = _DummyLifeOS(config={"cors": {"allowed_origins": origins}})

    app = create_app(life_os)

    mw = _cors_middleware(app)
    assert mw.kwargs["allow_origins"] == origins


def test_create_app_filters_blank_strings_from_allow_list():
    life_os = _DummyLifeOS(config={"cors": {"allowed_origins": ["http://localhost:8080", "   ", "*", 42]}})

    app = create_app(life_os)

    mw = _cors_middleware(app)
    assert mw.kwargs["allow_origins"] == ["http://localhost:8080"]


def test_create_app_tolerates_life_os_without_config_attribute():
    class Bare:
        pass

    app = create_app(Bare())

    mw = _cors_middleware(app)
    assert mw.kwargs["allow_origins"] == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


def test_create_app_registers_now_router():
    """Factory mounts the Week 8 Now-tab router.

    Earlier phases asserted *no* custom routes were wired; once the
    Now router lands it becomes the baseline assertion. Subsequent
    Week-8 tasks will add to this set (`/api/you`, `/api/people`, …).
    """
    app = create_app()

    custom_paths = {r.path for r in app.routes}
    expected_now_paths = {
        "/api/now",
        "/api/moments/{moment_id}/accept",
        "/api/moments/{moment_id}/dismiss",
        "/api/moments/{moment_id}/snooze",
        "/api/moments/{moment_id}/edit",
    }
    assert expected_now_paths.issubset(custom_paths)


def test_create_app_registers_you_and_people_routers():
    """Factory mounts the Week 8 You + People routers."""
    app = create_app()

    custom_paths = {r.path for r in app.routes}
    assert {"/api/you", "/api/people", "/api/people/{contact_id}"}.issubset(custom_paths)
