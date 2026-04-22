"""Smoke tests for ``scripts/measure_ollama_budget.py``.

Exercises the pure helpers (stats, cosine, report rendering, skip path) without
requiring a running Ollama server.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    """Load the script as a module despite the hyphenated-style script path."""
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "measure_ollama_budget.py"
    spec = importlib.util.spec_from_file_location("measure_ollama_budget", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_ollama_budget"] = mod
    spec.loader.exec_module(mod)
    return mod


mob = _load_module()


# ---------------------------------------------------------------------------
# OperationResult statistics
# ---------------------------------------------------------------------------


def test_operation_result_quantiles_single_sample():
    r = mob.OperationResult(name="x")
    r.samples.append(mob.Sample(latency_seconds=0.5))
    assert r.quantile(0.50) == 0.5
    assert r.quantile(0.95) == 0.5


def test_operation_result_quantiles_multi_sample():
    r = mob.OperationResult(name="x")
    for i in range(1, 11):
        r.samples.append(mob.Sample(latency_seconds=i / 10))
    # p50 should sit near the middle of 0.1..1.0
    p50 = r.quantile(0.50)
    assert p50 is not None
    assert 0.4 <= p50 <= 0.7
    # p99 should be near the top
    p99 = r.quantile(0.99)
    assert p99 is not None
    assert p99 >= 0.9


def test_operation_result_empty_returns_none():
    r = mob.OperationResult(name="empty")
    assert r.quantile(0.5) is None
    assert r.mean_tokens("prompt_tokens") is None


def test_mean_tokens_ignores_none():
    r = mob.OperationResult(name="t")
    r.samples.append(mob.Sample(latency_seconds=1.0, prompt_tokens=100))
    r.samples.append(mob.Sample(latency_seconds=1.0, prompt_tokens=None))
    r.samples.append(mob.Sample(latency_seconds=1.0, prompt_tokens=200))
    assert r.mean_tokens("prompt_tokens") == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def test_cosine_identity():
    v = [1.0, 2.0, 3.0]
    assert mob.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert mob.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    assert mob.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_mismatched_lengths_returns_zero():
    assert mob.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# sample_from_chat_response
# ---------------------------------------------------------------------------


def test_sample_from_chat_response_with_tokens():
    s = mob.sample_from_chat_response(
        latency=0.42,
        body={"prompt_eval_count": 100, "eval_count": 50, "message": {"content": "ok"}},
    )
    assert s.latency_seconds == 0.42
    assert s.prompt_tokens == 100
    assert s.completion_tokens == 50
    assert s.total_tokens == 150


def test_sample_from_chat_response_missing_tokens():
    s = mob.sample_from_chat_response(latency=1.0, body={})
    assert s.latency_seconds == 1.0
    assert s.prompt_tokens is None
    assert s.completion_tokens is None
    assert s.total_tokens is None


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_render_report_contains_all_operations():
    results = [
        mob.OperationResult(name="briefing_synthesis", samples=[mob.Sample(1.5, 100, 50, 150)]),
        mob.OperationResult(name="task_extraction", samples=[mob.Sample(0.9, 80, 20, 100)]),
    ]
    out = mob.render_report(
        results,
        chat_model="mistral",
        embed_model="nomic-embed-text",
        iterations=10,
        host_note="test",
    )
    assert "briefing_synthesis" in out
    assert "task_extraction" in out
    assert "mistral" in out
    assert "| Operation |" in out
    assert "_No errors recorded._" in out


def test_render_report_includes_errors_when_present():
    r = mob.OperationResult(name="boom")
    r.errors.append("Connection refused")
    out = mob.render_report([r], chat_model="m", embed_model="e", iterations=1, host_note="t")
    assert "Connection refused" in out
    assert "### boom" in out


def test_render_skip_report():
    out = mob.render_skip_report("http://localhost:11434", "reason here")
    assert "NOT RUN" in out
    assert "reason here" in out
    assert "Mac Mini" in out


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    ns = mob.parse_args([])
    assert ns.url == "http://localhost:11434"
    assert ns.chat_model == "mistral"
    assert ns.embed_model == "nomic-embed-text"
    assert ns.iterations == 10


def test_parse_args_overrides():
    ns = mob.parse_args(["--iterations", "3", "--chat-model", "phi3"])
    assert ns.iterations == 3
    assert ns.chat_model == "phi3"
