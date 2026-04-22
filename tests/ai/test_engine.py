"""Tests for :mod:`ai.engine`.

Strategy
--------
We mock the two HTTP-facing helpers (``_query_local`` / ``_query_cloud``)
so the tests never touch a live Ollama or Anthropic endpoint. Each
public method is exercised:

- **Dispatch** — ``draft_reply`` takes the cloud path iff
  ``use_cloud=True`` + ``cloud_api_key`` set; ``briefing_synthesis``,
  ``task_extraction``, ``priority_classification`` stay on Ollama
  regardless of ``use_cloud``.
- **Parsing** — ``task_extraction`` strips ```` ```json ```` fences,
  drops malformed JSON, rejects non-list / non-dict-element output.
  ``priority_classification`` coerces anything outside the four-word
  set to ``"normal"``.
- **Budgets** — every method raises :class:`AIBudgetExceeded` when its
  inner query sleeps past the per-instance budget. Budget attributes
  are class constants so the tests override them on the instance.
- **Semantic search** — returns ``[]`` when no vector store is wired;
  collapses chunk suffixes; dedupes by event id keeping best score;
  limits to ``k``; ignores malformed hits; enforces its own budget.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai.engine import (
    AIBudgetExceeded,
    AIEngine,
    SearchResult,
    _normalise_search_hits,
    _parse_task_list,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_engine(**kwargs: Any) -> AIEngine:
    defaults: dict[str, Any] = {
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }
    defaults.update(kwargs)
    return AIEngine(**defaults)


class _StubQueries:
    """Records calls to _query_local / _query_cloud and returns canned text.

    The engine dispatches through asyncio.wait_for, so the stubs are
    async. A per-stub ``delay`` lets tests simulate slow responses to
    drive :class:`AIBudgetExceeded`.
    """

    def __init__(
        self,
        local_response: str = "",
        cloud_response: str = "",
        local_delay: float = 0.0,
        cloud_delay: float = 0.0,
    ) -> None:
        self.local_response = local_response
        self.cloud_response = cloud_response
        self.local_delay = local_delay
        self.cloud_delay = cloud_delay
        self.local_calls: list[tuple[str, str]] = []
        self.cloud_calls: list[tuple[str, str]] = []

    async def local(self, system_prompt: str, user_prompt: str) -> str:
        self.local_calls.append((system_prompt, user_prompt))
        if self.local_delay:
            await asyncio.sleep(self.local_delay)
        return self.local_response

    async def cloud(self, system_prompt: str, user_prompt: str) -> str:
        self.cloud_calls.append((system_prompt, user_prompt))
        if self.cloud_delay:
            await asyncio.sleep(self.cloud_delay)
        return self.cloud_response


def _wire(engine: AIEngine, stubs: _StubQueries) -> None:
    engine._query_local = stubs.local  # type: ignore[method-assign]
    engine._query_cloud = stubs.cloud  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_use_cloud_requires_both_flag_and_key() -> None:
    # Flag true but no key → cloud disabled
    assert _make_engine(use_cloud=True).use_cloud is False
    # Key set but flag false → cloud disabled
    assert _make_engine(cloud_api_key="k", use_cloud=False).use_cloud is False
    # Both present → cloud enabled
    assert _make_engine(cloud_api_key="k", use_cloud=True).use_cloud is True


def test_ollama_url_trailing_slash_stripped() -> None:
    engine = _make_engine(ollama_url="http://localhost:11434/")
    assert engine.ollama_url == "http://localhost:11434"


def test_default_cloud_model_matches_ceo_plan() -> None:
    engine = _make_engine(cloud_api_key="k", use_cloud=True)
    assert engine.cloud_model == AIEngine.DEFAULT_CLOUD_MODEL


# ---------------------------------------------------------------------------
# briefing_synthesis
# ---------------------------------------------------------------------------


async def test_briefing_synthesis_uses_local_even_when_cloud_enabled() -> None:
    engine = _make_engine(cloud_api_key="k", use_cloud=True)
    stubs = _StubQueries(local_response="good morning", cloud_response="SHOULD NOT FIRE")
    _wire(engine, stubs)
    out = await engine.briefing_synthesis({"tasks": []})
    assert out == "good morning"
    assert len(stubs.local_calls) == 1
    assert stubs.cloud_calls == []


async def test_briefing_synthesis_serialises_context() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="ok")
    _wire(engine, stubs)
    await engine.briefing_synthesis({"tasks": [{"id": "t1"}], "unread": 3})
    _, user = stubs.local_calls[0]
    assert '"tasks"' in user and '"t1"' in user and '"unread": 3' in user


async def test_briefing_synthesis_budget_exceeded() -> None:
    engine = _make_engine()
    engine.BUDGET_BRIEFING = 0.05
    stubs = _StubQueries(local_response="late", local_delay=0.5)
    _wire(engine, stubs)
    with pytest.raises(AIBudgetExceeded) as excinfo:
        await engine.briefing_synthesis({})
    assert excinfo.value.operation == "briefing_synthesis"
    assert excinfo.value.budget_seconds == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# task_extraction
# ---------------------------------------------------------------------------


async def test_task_extraction_happy_path() -> None:
    engine = _make_engine()
    stubs = _StubQueries(
        local_response='[{"title": "Review PR", "due_hint": null, "priority": "high", "completed": false}]'
    )
    _wire(engine, stubs)
    event = {"payload": {"subject": "PR ready", "snippet": "please review"}}
    out = await engine.task_extraction(event)
    assert out == [{"title": "Review PR", "due_hint": None, "priority": "high", "completed": False}]


async def test_task_extraction_strips_markdown_fences() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response='```json\n[{"title": "X"}]\n```')
    _wire(engine, stubs)
    out = await engine.task_extraction({"payload": {"subject": "x"}})
    assert out == [{"title": "X"}]


async def test_task_extraction_returns_empty_on_malformed_json() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="not json at all")
    _wire(engine, stubs)
    assert await engine.task_extraction({"payload": {}}) == []


async def test_task_extraction_returns_empty_on_non_list() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response='{"not": "a list"}')
    _wire(engine, stubs)
    assert await engine.task_extraction({"payload": {}}) == []


async def test_task_extraction_drops_non_dict_elements() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response='[{"title": "keep"}, "junk", 42]')
    _wire(engine, stubs)
    assert await engine.task_extraction({"payload": {}}) == [{"title": "keep"}]


async def test_task_extraction_returns_empty_on_empty_array() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="[]")
    _wire(engine, stubs)
    assert await engine.task_extraction({"payload": {"subject": "50% off!"}}) == []


async def test_task_extraction_budget_exceeded() -> None:
    engine = _make_engine()
    engine.BUDGET_TASK_EXTRACTION = 0.05
    stubs = _StubQueries(local_response="[]", local_delay=0.5)
    _wire(engine, stubs)
    with pytest.raises(AIBudgetExceeded) as excinfo:
        await engine.task_extraction({"payload": {}})
    assert excinfo.value.operation == "task_extraction"


# ---------------------------------------------------------------------------
# priority_classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("critical", "critical"),
        ("High", "high"),
        ("  LOW  ", "low"),
        ("normal.", "normal"),  # stripped-trailing-dot kept; still valid? No: normal. != normal
    ],
)
async def test_priority_classification_valid_words(raw: str, expected: str) -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response=raw)
    _wire(engine, stubs)
    out = await engine.priority_classification({"payload": {}})
    # Note: "normal." contains punctuation; first-word split yields "normal."
    # which is not in _ALLOWED_PRIORITIES → default "normal". That still
    # returns "normal", so the parametrize case remains correct.
    if raw.strip().lower() == "normal.":
        assert out == "normal"
    else:
        assert out == expected


async def test_priority_classification_default_on_gibberish() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="potato")
    _wire(engine, stubs)
    assert await engine.priority_classification({"payload": {}}) == "normal"


async def test_priority_classification_default_on_empty() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="   ")
    _wire(engine, stubs)
    assert await engine.priority_classification({"payload": {}}) == "normal"


async def test_priority_classification_reads_payload_fields() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="high")
    _wire(engine, stubs)
    await engine.priority_classification(
        {
            "payload": {
                "from_address": "boss@acme.com",
                "subject": "URGENT",
                "snippet": "need this yesterday",
            }
        }
    )
    _, user = stubs.local_calls[0]
    assert "boss@acme.com" in user
    assert "URGENT" in user
    assert "need this yesterday" in user


async def test_priority_classification_budget_exceeded() -> None:
    engine = _make_engine()
    engine.BUDGET_PRIORITY = 0.05
    stubs = _StubQueries(local_response="high", local_delay=0.5)
    _wire(engine, stubs)
    with pytest.raises(AIBudgetExceeded) as excinfo:
        await engine.priority_classification({"payload": {}})
    assert excinfo.value.operation == "priority_classification"


# ---------------------------------------------------------------------------
# draft_reply
# ---------------------------------------------------------------------------


async def test_draft_reply_routes_to_cloud_when_enabled() -> None:
    engine = _make_engine(cloud_api_key="k", use_cloud=True)
    stubs = _StubQueries(local_response="LOCAL", cloud_response="cloud draft")
    _wire(engine, stubs)
    out = await engine.draft_reply("alice", [{"direction": "inbound", "body": "hi"}], {})
    assert out == "cloud draft"
    assert stubs.cloud_calls and not stubs.local_calls


async def test_draft_reply_falls_back_to_local_without_cloud() -> None:
    engine = _make_engine()  # use_cloud=False
    stubs = _StubQueries(local_response="local draft", cloud_response="SHOULD NOT FIRE")
    _wire(engine, stubs)
    out = await engine.draft_reply("alice", [], {"formality": 0.2})
    assert out == "local draft"
    assert stubs.local_calls and not stubs.cloud_calls


async def test_draft_reply_budget_exceeded_on_cloud_path() -> None:
    engine = _make_engine(cloud_api_key="k", use_cloud=True)
    engine.BUDGET_DRAFT_REPLY = 0.05
    stubs = _StubQueries(cloud_response="late", cloud_delay=0.5)
    _wire(engine, stubs)
    with pytest.raises(AIBudgetExceeded) as excinfo:
        await engine.draft_reply("alice", [], {})
    assert excinfo.value.operation == "draft_reply"


async def test_draft_reply_includes_contact_and_style_in_prompt() -> None:
    engine = _make_engine()
    stubs = _StubQueries(local_response="ok")
    _wire(engine, stubs)
    await engine.draft_reply(
        "c-7",
        [{"direction": "inbound", "body": "proposal?"}],
        {"contact_name": "Alice", "formality": 0.3},
    )
    _, user = stubs.local_calls[0]
    assert "c-7" in user and "Alice" in user and "proposal?" in user


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------


class _FakeVectorStore:
    def __init__(self, hits: list[dict[str, Any]], delay: float = 0.0) -> None:
        self.hits = hits
        self.delay = delay
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        import time as _time

        self.calls.append((query, limit))
        if self.delay:
            _time.sleep(self.delay)
        return list(self.hits)


async def test_semantic_search_returns_empty_without_vector_store() -> None:
    engine = _make_engine()
    assert await engine.semantic_search("anything", k=5) == []


async def test_semantic_search_happy_path() -> None:
    store = _FakeVectorStore(
        [
            {
                "doc_id": "evt-1",
                "score": 0.9,
                "type": "email.received",
                "source": "proton_mail",
                "timestamp": 1111,
                "snippet": "hello",
            },
            {
                "doc_id": "evt-2",
                "score": 0.5,
                "type": "message.received",
                "source": "imessage",
                "timestamp": 2222,
                "snippet": "yo",
            },
        ]
    )
    engine = _make_engine(vector_store=store)
    out = await engine.semantic_search("greeting", k=5)
    assert [h.event_id for h in out] == ["evt-1", "evt-2"]
    assert out[0].score == pytest.approx(0.9)
    assert out[0].event_type == "email.received"
    assert out[0].source == "proton_mail"
    assert out[0].timestamp == 1111


async def test_semantic_search_collapses_chunk_suffixes_best_score_wins() -> None:
    store = _FakeVectorStore(
        [
            {"doc_id": "evt-1_0", "score": 0.4},
            {"doc_id": "evt-1_1", "score": 0.8},
            {"doc_id": "evt-1_2", "score": 0.6},
        ]
    )
    engine = _make_engine(vector_store=store)
    out = await engine.semantic_search("q", k=5)
    assert len(out) == 1
    assert out[0].event_id == "evt-1"
    assert out[0].score == pytest.approx(0.8)


async def test_semantic_search_respects_k_limit() -> None:
    store = _FakeVectorStore([{"doc_id": f"evt-{i}", "score": 1.0 - i * 0.1} for i in range(10)])
    engine = _make_engine(vector_store=store)
    out = await engine.semantic_search("q", k=3)
    assert len(out) == 3
    assert [h.event_id for h in out] == ["evt-0", "evt-1", "evt-2"]


async def test_semantic_search_k_zero_coerced_to_one() -> None:
    store = _FakeVectorStore([{"doc_id": "evt-1", "score": 0.5}])
    engine = _make_engine(vector_store=store)
    out = await engine.semantic_search("q", k=0)
    assert len(out) == 1


async def test_semantic_search_ignores_malformed_hits() -> None:
    store = _FakeVectorStore(
        [
            {"doc_id": "evt-good", "score": 0.7},
            "junk",  # not a dict
            {"doc_id": ""},  # blank id
            {"score": 0.9},  # missing id
        ]
    )
    engine = _make_engine(vector_store=store)
    out = await engine.semantic_search("q", k=5)
    assert [h.event_id for h in out] == ["evt-good"]


async def test_semantic_search_budget_exceeded() -> None:
    store = _FakeVectorStore([{"doc_id": "evt-1", "score": 0.9}], delay=0.5)
    engine = _make_engine(vector_store=store)
    engine.BUDGET_SEMANTIC_SEARCH = 0.05
    with pytest.raises(AIBudgetExceeded) as excinfo:
        await engine.semantic_search("slow", k=5)
    assert excinfo.value.operation == "semantic_search"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_task_list_strips_json_fence() -> None:
    assert _parse_task_list('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_parse_task_list_strips_bare_fence() -> None:
    assert _parse_task_list('```\n[{"a": 1}]\n```') == [{"a": 1}]


def test_parse_task_list_empty_string() -> None:
    assert _parse_task_list("") == []


def test_parse_task_list_whitespace_only() -> None:
    assert _parse_task_list("   \n\n  ") == []


def test_normalise_search_hits_returns_sorted_desc() -> None:
    hits = [
        {"doc_id": "a", "score": 0.3},
        {"doc_id": "b", "score": 0.9},
        {"doc_id": "c", "score": 0.5},
    ]
    out = _normalise_search_hits(hits, k=5)
    assert [r.event_id for r in out] == ["b", "c", "a"]


def test_normalise_search_hits_none_score_coerces_zero() -> None:
    out = _normalise_search_hits([{"doc_id": "x", "score": None}], k=5)
    assert out == [SearchResult(event_id="x", event_type="", source="", timestamp=0, snippet="", score=0.0)]


def test_search_result_is_frozen() -> None:
    r = SearchResult(event_id="x", event_type="t", source="s", timestamp=1, snippet="p", score=0.5)
    with pytest.raises(AttributeError):
        r.score = 0.9  # type: ignore[misc]
