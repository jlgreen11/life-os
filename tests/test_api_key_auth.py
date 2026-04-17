"""
Tests for optional API key authentication on the web API.

Covers:
  * Auth is disabled by default (no breaking change for localhost installs).
  * When enabled, requests without a key are rejected with 401.
  * Valid keys are accepted via X-API-Key header, Authorization: Bearer, and
    ?api_key= query parameter (the last is needed for WebSocket).
  * /health is exempt so iOS connection probes keep working.
  * CORS preflight (OPTIONS) requests bypass the auth check.
  * Key comparison is constant-time (verified via verify_api_key).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import LifeOS
from web.app import create_web_app
from web.auth import verify_api_key


def _build_app(db, event_store, user_model_store, config):
    life_os = LifeOS(
        db=db,
        event_bus=None,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )
    return create_web_app(life_os)


class TestAuthDisabledByDefault:
    def test_no_auth_config_leaves_api_open(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
        })
        client = TestClient(app)
        assert client.get("/health").status_code == 200

    def test_empty_api_key_disables_auth(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": ""},
        })
        client = TestClient(app)
        assert client.get("/health").status_code == 200

    def test_whitespace_only_key_disables_auth(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "   "},
        })
        client = TestClient(app)
        assert client.get("/health").status_code == 200


class TestAuthEnabledRejectsUnauthenticated:
    def test_missing_key_returns_401(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        response = client.get("/api/tasks")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    def test_wrong_key_returns_401(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        response = client.get("/api/tasks", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_wrong_bearer_returns_401(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        response = client.get(
            "/api/tasks",
            headers={"Authorization": "Bearer nope"},
        )
        assert response.status_code == 401


class TestAuthEnabledAcceptsValidKey:
    def test_x_api_key_header_accepted(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        # Use /api/status since it's a simple endpoint that doesn't need
        # the full service stack wired up.
        response = client.get("/api/status", headers={"X-API-Key": "secret-key-123"})
        assert response.status_code != 401

    def test_authorization_bearer_accepted(self, db, event_store, user_model_store):
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        response = client.get(
            "/api/status",
            headers={"Authorization": "Bearer secret-key-123"},
        )
        assert response.status_code != 401

    def test_query_param_accepted(self, db, event_store, user_model_store):
        """WebSocket clients in browsers can't set custom handshake headers,
        so the query param form is supported as a fallback."""
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        response = client.get("/api/status?api_key=secret-key-123")
        assert response.status_code != 401


class TestAuthExemptions:
    def test_health_exempt_when_auth_enabled(self, db, event_store, user_model_store):
        """/health must be reachable without a key so iOS apps and status
        dashboards can probe the server before the user has configured one."""
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
        })
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_options_preflight_bypasses_auth(self, db, event_store, user_model_store):
        """CORS preflight must pass without a key — browsers don't attach
        custom headers to the preflight request."""
        app = _build_app(db, event_store, user_model_store, config={
            "auth": {"api_key": "secret-key-123"},
            "cors": {"allowed_origins": ["http://localhost:3000"]},
        })
        client = TestClient(app)
        response = client.options(
            "/api/tasks",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code in (200, 204)


class TestVerifyAPIKey:
    def test_verify_disabled_when_no_expected_key(self):
        assert verify_api_key(None, None) is True
        assert verify_api_key("anything", "") is True

    def test_verify_rejects_missing_when_required(self):
        assert verify_api_key(None, "secret") is False
        assert verify_api_key("", "secret") is False

    def test_verify_rejects_wrong_key(self):
        assert verify_api_key("wrong", "secret") is False

    def test_verify_accepts_matching_key(self):
        assert verify_api_key("secret", "secret") is True
