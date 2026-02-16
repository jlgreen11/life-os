"""
Comprehensive test suite for CORS (Cross-Origin Resource Sharing) security.

CORS controls which domains can make authenticated requests to the Life OS API.
Misconfigured CORS is a critical security vulnerability that can leak sensitive
personal data (emails, calendar, messages, transactions) to malicious websites.

This test suite verifies:
1. Default secure behavior (localhost-only when config is missing)
2. Proper parsing and validation of allowed_origins from config
3. Rejection of requests from unauthorized origins
4. Acceptance of requests from explicitly allowed origins
5. Handling of edge cases (empty lists, invalid types, malformed values)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import LifeOS
from web.app import create_web_app


class TestCORSDefaultBehavior:
    """Test CORS security when no configuration is provided."""

    def test_missing_cors_config_defaults_to_localhost(self, db, event_store, user_model_store):
        """
        When the 'cors' key is missing from config, the application should
        default to a secure localhost-only policy.

        This ensures that misconfigured or minimal deployments are secure by
        default and cannot accidentally expose the API to arbitrary origins.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            # No 'cors' key → should default to localhost-only
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        # The CORS middleware is the first (and only) user middleware.
        # We access its options via the Middleware wrapper object.
        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        # Verify default origins are localhost-only
        assert "http://localhost:8080" in options["allow_origins"]
        assert "http://127.0.0.1:8080" in options["allow_origins"]
        # Should NOT include wildcard
        assert "*" not in options["allow_origins"]

    def test_empty_cors_config_defaults_to_localhost(self, db, event_store, user_model_store):
        """
        When the 'cors' key exists but is empty, the application should
        default to a secure localhost-only policy.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {},  # Empty dict
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        assert "http://localhost:8080" in options["allow_origins"]
        assert "http://127.0.0.1:8080" in options["allow_origins"]

    def test_none_allowed_origins_defaults_to_localhost(self, db, event_store, user_model_store):
        """
        When allowed_origins is explicitly set to None, the application
        should default to a secure localhost-only policy.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": None,
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        assert "http://localhost:8080" in options["allow_origins"]
        assert "http://127.0.0.1:8080" in options["allow_origins"]


class TestCORSConfigValidation:
    """Test CORS configuration parsing and validation."""

    def test_valid_allowed_origins_list(self, db, event_store, user_model_store):
        """
        When a valid list of allowed origins is provided in the config,
        those origins should be used exactly as specified.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                    "https://myapp.example.com",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        assert "http://localhost:3000" in options["allow_origins"]
        assert "https://myapp.example.com" in options["allow_origins"]
        # Should NOT include the default localhost:8080 (user config overrides)
        assert "http://localhost:8080" not in options["allow_origins"]

    def test_empty_list_defaults_to_localhost(self, db, event_store, user_model_store):
        """
        When allowed_origins is an empty list, the application should default
        to localhost-only for security. An empty list likely indicates a
        configuration error, not an intentional lockdown.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        assert "http://localhost:8080" in options["allow_origins"]
        assert "http://127.0.0.1:8080" in options["allow_origins"]

    def test_invalid_type_defaults_to_localhost(self, db, event_store, user_model_store):
        """
        When allowed_origins is not a list (e.g., a string or number), the
        application should default to localhost-only for security.
        """
        for invalid_value in ["http://localhost:3000", 123, True, {"origin": "test"}]:
            config = {
                "data_dir": "./data",
                "nats_url": "nats://localhost:4222",
                "cors": {
                    "allowed_origins": invalid_value,
                },
            }

            life_os = LifeOS(
                db=db,
                event_bus=None,
                event_store=event_store,
                user_model_store=user_model_store,
                config=config,
            )
            app = create_web_app(life_os)

            cors_middleware = app.user_middleware[0]
            options = cors_middleware.kwargs

            assert "http://localhost:8080" in options["allow_origins"]
            assert "http://127.0.0.1:8080" in options["allow_origins"]

    def test_filters_out_empty_strings(self, db, event_store, user_model_store):
        """
        Empty strings in the allowed_origins list should be filtered out.
        If the list becomes empty after filtering, default to localhost-only.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": ["", "  ", "\t"],  # All whitespace
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        # Should fall back to localhost-only
        assert "http://localhost:8080" in options["allow_origins"]
        assert "http://127.0.0.1:8080" in options["allow_origins"]

    def test_filters_out_non_strings(self, db, event_store, user_model_store):
        """
        Non-string values in the allowed_origins list should be filtered out.
        Only valid string origins should remain.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",  # Valid
                    None,  # Invalid
                    123,  # Invalid
                    "https://example.com",  # Valid
                    False,  # Invalid
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        # Only the valid strings should remain
        assert "http://localhost:3000" in options["allow_origins"]
        assert "https://example.com" in options["allow_origins"]
        # Invalid values should be filtered out
        assert None not in options["allow_origins"]
        assert 123 not in options["allow_origins"]
        assert False not in options["allow_origins"]

    def test_strips_whitespace_from_origins(self, db, event_store, user_model_store):
        """
        Leading and trailing whitespace in origin URLs should be stripped.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "  http://localhost:3000  ",
                    "\thttps://example.com\n",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)

        cors_middleware = app.user_middleware[0]
        options = cors_middleware.kwargs

        # Whitespace should be stripped
        assert "http://localhost:3000" in options["allow_origins"]
        assert "https://example.com" in options["allow_origins"]


class TestCORSRequestHandling:
    """Test actual HTTP requests with various Origin headers."""

    def test_request_from_allowed_origin_succeeds(self, db, event_store, user_model_store):
        """
        Requests from an explicitly allowed origin should include the
        Access-Control-Allow-Origin header in the response.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # Make a preflight request (OPTIONS) from the allowed origin
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # The response should include the CORS header
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_request_from_disallowed_origin_rejected(self, db, event_store, user_model_store):
        """
        Requests from an origin NOT in the allowed list should NOT receive
        the Access-Control-Allow-Origin header, which causes the browser to
        block the request.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # Make a preflight request from a disallowed origin
        response = client.options(
            "/health",
            headers={"Origin": "https://evil.com"},
        )

        # The response should NOT include the CORS header for this origin
        # (or if it does, it should be "null" or absent)
        cors_header = response.headers.get("access-control-allow-origin")
        assert cors_header != "https://evil.com"

    def test_request_without_origin_header_succeeds(self, db, event_store, user_model_store):
        """
        Requests without an Origin header (same-origin requests or direct
        API calls) should succeed regardless of CORS configuration.

        CORS only affects cross-origin browser requests. API calls from curl,
        Python scripts, or same-origin requests don't include Origin headers
        and are not subject to CORS restrictions.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # Make a request without Origin header (simulates same-origin or API call)
        response = client.get("/health")

        # Request should succeed (CORS doesn't block same-origin)
        assert response.status_code == 200

    def test_wildcard_origin_allows_all(self, db, event_store, user_model_store):
        """
        If the user explicitly configures the wildcard origin "*", requests
        from any origin should be allowed.

        WARNING: This is insecure and should only be used for local development
        or testing. Never use "*" in production.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": ["*"],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # Make requests from various origins
        for origin in ["http://localhost:3000", "https://evil.com", "https://example.com"]:
            response = client.options(
                "/health",
                headers={"Origin": origin},
            )

            # All should be allowed with wildcard
            # Note: Starlette's CORS middleware returns "*" for the allow-origin
            # header when configured with ["*"]
            cors_header = response.headers.get("access-control-allow-origin")
            assert cors_header == "*"


class TestCORSCredentials:
    """Test CORS handling of credentials (cookies, auth headers)."""

    def test_credentials_allowed_for_allowed_origins(self, db, event_store, user_model_store):
        """
        The Access-Control-Allow-Credentials header should be set to 'true'
        for requests from allowed origins, enabling authenticated requests.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # Credentials should be allowed
        assert response.headers.get("access-control-allow-credentials") == "true"


class TestCORSMethods:
    """Test CORS handling of HTTP methods."""

    def test_all_methods_allowed(self, db, event_store, user_model_store):
        """
        The Access-Control-Allow-Methods header should include all standard
        HTTP methods (GET, POST, PUT, DELETE, OPTIONS, etc.).
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # All methods should be allowed (FastAPI CORS middleware returns "*")
        allowed_methods = response.headers.get("access-control-allow-methods", "")
        # The response should include common methods
        for method in ["GET", "POST", "PUT", "DELETE"]:
            assert method in allowed_methods


class TestCORSSecurityScenarios:
    """Test real-world security scenarios."""

    def test_prevents_csrf_from_malicious_site(self, db, event_store, user_model_store):
        """
        Scenario: A user visits evil.com while logged into their Life OS
        instance. evil.com attempts to make an authenticated request to the
        Life OS API to steal personal data.

        Expected: The browser blocks the request because evil.com is not in
        the allowed_origins list.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:8080",  # Only the official UI
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # evil.com tries to fetch user notifications
        response = client.options(
            "/api/notifications",
            headers={"Origin": "https://evil.com"},
        )

        # The CORS header should NOT allow evil.com
        cors_header = response.headers.get("access-control-allow-origin")
        assert cors_header != "https://evil.com"

    def test_allows_legitimate_mobile_app(self, db, event_store, user_model_store):
        """
        Scenario: The user has a mobile app running on localhost:3000 during
        development. The app needs to make authenticated requests to the
        Life OS API.

        Expected: The request succeeds because localhost:3000 is explicitly
        allowed in the config.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "http://localhost:8080",
                    "http://localhost:3000",  # Mobile app dev server
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # Mobile app makes a preflight request
        response = client.options(
            "/api/tasks",
            headers={"Origin": "http://localhost:3000"},
        )

        # Should be allowed
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_production_deployment_locked_down(self, db, event_store, user_model_store):
        """
        Scenario: Production deployment with a single trusted frontend domain.

        Expected: Only the production frontend can make requests. All other
        origins are blocked.
        """
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "cors": {
                "allowed_origins": [
                    "https://mylifeos.example.com",  # Production frontend only
                ],
            },
        }

        life_os = LifeOS(
            db=db,
            event_bus=None,
            event_store=event_store,
            user_model_store=user_model_store,
            config=config,
        )
        app = create_web_app(life_os)
        client = TestClient(app)

        # Production frontend request should succeed
        prod_response = client.options(
            "/api/events",
            headers={"Origin": "https://mylifeos.example.com"},
        )
        assert prod_response.headers.get("access-control-allow-origin") == "https://mylifeos.example.com"

        # Request from localhost should fail (not in production allowed list)
        local_response = client.options(
            "/api/events",
            headers={"Origin": "http://localhost:8080"},
        )
        cors_header = local_response.headers.get("access-control-allow-origin")
        assert cors_header != "http://localhost:8080"
