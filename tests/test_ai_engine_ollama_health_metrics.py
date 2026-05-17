"""
Tests for AIEngine Ollama health metrics — rolling success rate, latency, and
outcome classification.

Coverage:
1. Successful call increments `requests_success` and pushes latency.
2. httpx.TimeoutException increments `requests_timeout` and records last_failure_at.
3. Generic ConnectError (failure) increments `requests_failed` and stores
   `last_error_message`.
4. Empty response body increments `requests_empty_response` without raising.
5. `rolling_success_rate` is bounded by the last 50 outcomes — older calls
   are evicted from the rolling window.
6. `get_ollama_health()` returns a JSON-serializable shape (no deque leakage).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from services.ai_engine.engine import AIEngine, AIEngineError


def _mock_ollama_response(content: str) -> Mock:
    """Build a mock httpx response that returns `content` from /api/chat."""
    mock_response_obj = Mock()
    mock_response_obj.json.return_value = {
        "message": {"role": "assistant", "content": content}
    }
    mock_response_obj.raise_for_status = Mock()
    return mock_response_obj


@pytest.mark.asyncio
async def test_successful_call_increments_success_and_latency(db, user_model_store):
    """A 200/non-empty response should bump success counters and record latency."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_ollama_response("hello"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_local("sys", "usr")

    assert result == "hello"
    health = engine.get_ollama_health()
    assert health["requests_total"] == 1
    assert health["requests_success"] == 1
    assert health["requests_failed"] == 0
    assert health["requests_empty_response"] == 0
    assert health["requests_timeout"] == 0
    assert len(health["latency_ms_window"]) == 1
    assert health["latency_ms_window"][0] >= 0
    assert health["recent_outcomes"] == ["success"]
    assert health["rolling_success_rate"] == 1.0
    assert health["last_success_at"] is not None
    assert health["last_failure_at"] is None


@pytest.mark.asyncio
async def test_timeout_exception_increments_timeout_counter(db, user_model_store):
    """httpx.TimeoutException should classify the outcome as 'timeout'."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("slow ollama"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(AIEngineError):
            await engine._query_local("sys", "usr")

    health = engine.get_ollama_health()
    assert health["requests_total"] == 1
    assert health["requests_timeout"] == 1
    assert health["requests_success"] == 0
    assert health["recent_outcomes"] == ["timeout"]
    assert health["last_failure_at"] is not None
    assert "slow ollama" in (health["last_error_message"] or "")
    assert health["rolling_success_rate"] == 0.0


@pytest.mark.asyncio
async def test_generic_failure_increments_failed_and_stores_message(db, user_model_store):
    """A ConnectError (non-timeout failure) is classified as 'fail'."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        with pytest.raises(AIEngineError):
            await engine._query_local("sys", "usr")

    health = engine.get_ollama_health()
    assert health["requests_total"] == 1
    assert health["requests_failed"] == 1
    assert health["requests_timeout"] == 0
    assert health["recent_outcomes"] == ["fail"]
    assert "refused" in (health["last_error_message"] or "")


@pytest.mark.asyncio
async def test_empty_response_increments_empty_counter(db, user_model_store):
    """A 200 with empty content body should be counted as 'empty', not 'success'."""
    engine = AIEngine(db, user_model_store, {})

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_ollama_response(""))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Empty response is not an exception — caller still gets "" back.
        result = await engine._query_local("sys", "usr")

    assert result == ""
    health = engine.get_ollama_health()
    assert health["requests_total"] == 1
    assert health["requests_empty_response"] == 1
    assert health["requests_success"] == 0
    assert health["recent_outcomes"] == ["empty"]
    assert health["rolling_success_rate"] == 0.0
    # Latency is recorded because the HTTP request actually completed.
    assert len(health["latency_ms_window"]) == 1


@pytest.mark.asyncio
async def test_rolling_success_rate_excludes_data_older_than_last_50(db, user_model_store):
    """The rolling window must drop outcomes older than the most recent 50 calls."""
    engine = AIEngine(db, user_model_store, {})

    # Stage 1: 50 failures so the entire rolling window is "fail".
    for _ in range(50):
        engine._record_ollama_outcome("fail", error_message="x")
        engine._ollama_metrics["requests_total"] += 1
    pre = engine.get_ollama_health()
    assert pre["rolling_success_rate"] == 0.0
    assert len(pre["recent_outcomes"]) == 50

    # Stage 2: 50 successes — should fully evict the earlier failures from the
    # rolling deque (maxlen=50), driving rolling_success_rate to 1.0 even
    # though cumulative `requests_failed` is still 50.
    for _ in range(50):
        engine._record_ollama_outcome("success")
        engine._ollama_metrics["requests_total"] += 1
    post = engine.get_ollama_health()
    assert post["rolling_success_rate"] == 1.0
    assert len(post["recent_outcomes"]) == 50
    assert all(o == "success" for o in post["recent_outcomes"])
    # Cumulative counters are NOT reset — they track lifetime totals.
    assert post["requests_failed"] == 50
    assert post["requests_success"] == 50
    assert post["requests_total"] == 100


def test_get_ollama_health_returns_json_serializable_shape(db, user_model_store):
    """The health snapshot must be directly JSON-encodable (no deque/datetime leakage)."""
    engine = AIEngine(db, user_model_store, {})
    # Populate at least one outcome of each kind to exercise all fields.
    engine._ollama_metrics["requests_total"] = 4
    engine._record_ollama_outcome("success")
    engine._record_ollama_outcome("empty", error_message="nothing")
    engine._record_ollama_outcome("timeout", error_message="slow")
    engine._record_ollama_outcome("fail", error_message="boom")
    engine._ollama_metrics["latency_ms_window"].append(12.5)

    health = engine.get_ollama_health()
    # json.dumps will raise TypeError on un-serializable types (deque, datetime).
    encoded = json.dumps(health)
    decoded = json.loads(encoded)

    # Spot-check the expected keys survive round-trip.
    assert decoded["requests_success"] == 1
    assert decoded["requests_empty_response"] == 1
    assert decoded["requests_timeout"] == 1
    assert decoded["requests_failed"] == 1
    assert decoded["latency_ms_window"] == [12.5]
    assert decoded["median_latency_ms"] == 12.5
    assert decoded["rolling_success_rate"] == pytest.approx(0.25)
    assert decoded["last_error_message"] == "boom"


def test_get_ollama_health_with_no_calls_returns_safe_defaults(db, user_model_store):
    """Before any calls, derived metrics should be None rather than crashing."""
    engine = AIEngine(db, user_model_store, {})
    health = engine.get_ollama_health()
    assert health["requests_total"] == 0
    assert health["rolling_success_rate"] is None
    assert health["median_latency_ms"] is None
    assert health["recent_outcomes"] == []
    assert health["latency_ms_window"] == []
    assert health["last_success_at"] is None
    assert health["last_failure_at"] is None
