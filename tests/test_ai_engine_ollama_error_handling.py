"""
Tests for AIEngine Ollama error handling — structured error classification.

Verifies that _query_local() catches httpx exceptions and raises AIEngineError
with machine-readable error_type ('connection', 'timeout', 'server_error')
so callers and operators can diagnose failures without reading raw stack traces.

Coverage:
1. Connection refused → AIEngineError(error_type='connection')
2. Request timeout → AIEngineError(error_type='timeout')
3. HTTP 500 → AIEngineError(error_type='server_error')
4. Successful response → normal string return (no regression)
5. generate_briefing catches AIEngineError and returns diagnostic message
"""

import httpx
import pytest
from unittest.mock import AsyncMock, Mock, patch

from services.ai_engine.engine import AIEngine, AIEngineError


# -------------------------------------------------------------------
# AIEngineError unit tests
# -------------------------------------------------------------------


def test_ai_engine_error_attributes():
    """AIEngineError stores error_type and details for programmatic access."""
    err = AIEngineError("something broke", "connection", "details here")
    assert str(err) == "something broke"
    assert err.error_type == "connection"
    assert err.details == "details here"


def test_ai_engine_error_default_details():
    """AIEngineError defaults details to empty string."""
    err = AIEngineError("fail", "timeout")
    assert err.details == ""


def test_ai_engine_error_is_exception():
    """AIEngineError is a proper Exception subclass (caught by bare except)."""
    err = AIEngineError("msg", "connection")
    assert isinstance(err, Exception)


# -------------------------------------------------------------------
# _query_local error classification tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_local_connect_error_raises_connection(db, user_model_store):
    """ConnectError from httpx should become AIEngineError(error_type='connection')."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(AIEngineError) as exc_info:
            await engine._query_local("sys", "usr")

        assert exc_info.value.error_type == "connection"
        assert "unreachable" in str(exc_info.value).lower()
        assert "localhost:11434" in exc_info.value.details


@pytest.mark.asyncio
async def test_query_local_timeout_raises_timeout(db, user_model_store):
    """TimeoutException from httpx should become AIEngineError(error_type='timeout')."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(AIEngineError) as exc_info:
            await engine._query_local("sys", "usr")

        assert exc_info.value.error_type == "timeout"
        assert "timed out" in str(exc_info.value).lower()
        assert "120s" in exc_info.value.details


@pytest.mark.asyncio
async def test_query_local_http_500_raises_server_error(db, user_model_store):
    """HTTP 500 from Ollama should become AIEngineError(error_type='server_error')."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        # Build a realistic HTTPStatusError with a mock response
        mock_request = httpx.Request("POST", "http://localhost:11434/api/chat")
        mock_response = httpx.Response(500, request=mock_request)
        mock_response_obj = Mock()
        mock_response_obj.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mock_request, response=mock_response
        )
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(AIEngineError) as exc_info:
            await engine._query_local("sys", "usr")

        assert exc_info.value.error_type == "server_error"
        assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_query_local_success_returns_string(db, user_model_store):
    """Successful Ollama response should return the content string (no regression)."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = {
            "message": {"role": "assistant", "content": "Hello from Ollama"}
        }
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_local("sys", "usr")
        assert result == "Hello from Ollama"


@pytest.mark.asyncio
async def test_query_local_custom_url_in_connection_error(db, user_model_store):
    """Connection error details should reference the configured Ollama URL."""
    config = {"ollama_url": "http://custom-host:9999"}
    engine = AIEngine(db, user_model_store, config)

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(AIEngineError) as exc_info:
            await engine._query_local("sys", "usr")

        assert "custom-host:9999" in exc_info.value.details


# -------------------------------------------------------------------
# generate_briefing error handling tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_briefing_catches_ai_engine_error(db, user_model_store):
    """generate_briefing should catch AIEngineError and return a diagnostic message."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context") as mock_ctx:
        mock_ctx.return_value = "context"

        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.side_effect = AIEngineError(
                "Ollama service unreachable", "connection", "Could not connect to http://localhost:11434"
            )

            result = await engine.generate_briefing()

            # Should not raise — returns a diagnostic string instead
            assert "unavailable" in result.lower()
            assert "connection" in result.lower()
            assert "localhost:11434" in result


@pytest.mark.asyncio
async def test_generate_briefing_catches_timeout_error(db, user_model_store):
    """generate_briefing should surface timeout diagnostics to the caller."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context") as mock_ctx:
        mock_ctx.return_value = "context"

        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.side_effect = AIEngineError(
                "Ollama request timed out", "timeout", "Request exceeded 120s timeout"
            )

            result = await engine.generate_briefing()

            assert "timed out" in result.lower()
            assert "timeout" in result.lower()


@pytest.mark.asyncio
async def test_generate_briefing_success_unaffected(db, user_model_store):
    """generate_briefing should return normal response when no error occurs."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context") as mock_ctx:
        mock_ctx.return_value = "context"

        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Good morning! Here's your briefing."

            result = await engine.generate_briefing()
            assert result == "Good morning! Here's your briefing."
