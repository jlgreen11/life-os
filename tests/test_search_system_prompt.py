"""
Tests for the enhanced search_life() system prompt.

The system prompt was expanded from 3 generic lines to a multi-section
synthesis guide that instructs the LLM on how to use the 5-section search
context assembled by ContextAssembler.assemble_search_context():

  Section 1 — Search intent (the raw query)
  Section 2 — Current timestamp (for relative-date resolution)
  Section 3 — User preferences (verbosity, name)
  Section 4 — Known semantic facts (disambiguation of ambiguous references)
  Section 5 — Recent mood signals (tone calibration)
  + Appended  — JSON search results ranked by vector similarity

Without this guide the LLM tends to enumerate results chronologically or
hallucinate connections between unrelated events rather than synthesising a
coherent, grounded answer.

Coverage:
 1. Prompt contains temporal reasoning guidance (relative date translation)
 2. Prompt instructs use of Known facts for disambiguation
 3. Prompt requires first-sentence direct answer (no query restatement)
 4. Prompt specifies citation format (date + source type + content)
 5. Prompt gives no-result handling instructions
 6. Prompt includes anti-hallucination constraint
 7. Prompt instructs plain prose (no unnecessary bullet points)
 8. Prompt references mood context for brevity under high stress
 9. search_life() passes the enhanced prompt to _query_local
10. Return value from the model is forwarded unchanged
11. Prompt contains disambiguation section
12. Prompt instructs grouping related results thematically
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from services.ai_engine.engine import AIEngine


# ---------------------------------------------------------------------------
# Helper: capture the system prompt actually passed to _query_local
# ---------------------------------------------------------------------------

async def _capture_search_prompt(db, user_model_store, query: str = "test query") -> str:
    """
    Invoke search_life() with mocked context and query layers and return the
    system prompt that was actually passed to _query_local.

    Args:
        db: DatabaseManager fixture (provides real SQLite connections).
        user_model_store: UserModelStore fixture.
        query: The search query string to pass through.

    Returns:
        The system prompt string captured from the _query_local call.
    """
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_search_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Found 2 results."
            engine.vector_store = None  # disable real vector search
            await engine.search_life(query)

    return mock_local.call_args[0][0]


# ---------------------------------------------------------------------------
# 1. Temporal reasoning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_contains_temporal_reasoning(db, user_model_store):
    """Prompt must instruct the LLM to resolve relative date expressions."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "temporal" in prompt.lower() or "current time" in prompt.lower(), \
        "Prompt must reference temporal reasoning or the current time section"
    assert any(phrase in prompt.lower() for phrase in ("last month", "yesterday", "last week")), \
        "Prompt must give examples of relative date expressions to resolve"


# ---------------------------------------------------------------------------
# 2. Fact-based disambiguation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_contains_disambiguation_guidance(db, user_model_store):
    """Prompt must instruct LLM to use Known facts to resolve ambiguous references."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "disamb" in prompt.lower() or "known facts" in prompt.lower(), \
        "Prompt must reference disambiguation using known facts"


# ---------------------------------------------------------------------------
# 3. Direct first-sentence answer requirement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_requires_direct_answer_first(db, user_model_store):
    """Prompt must instruct LLM to answer directly without restating the query."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "first sentence" in prompt.lower() or "directly" in prompt.lower(), \
        "Prompt must require a direct answer in the first sentence"
    assert "restat" in prompt.lower() or "rephrase" in prompt.lower(), \
        "Prompt must forbid restating/rephrasing the query"


# ---------------------------------------------------------------------------
# 4. Citation format requirements
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_specifies_citation_format(db, user_model_store):
    """Prompt must specify date + source type + content citation format."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "citation" in prompt.lower() or "cite" in prompt.lower(), \
        "Prompt must specify a citation format"
    assert "date" in prompt.lower(), \
        "Citation format must require dates"
    assert "source" in prompt.lower(), \
        "Citation format must require source type attribution"


# ---------------------------------------------------------------------------
# 5. No-result handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_contains_no_result_handling(db, user_model_store):
    """Prompt must give explicit instructions for when no results match."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "no" in prompt.lower() and ("result" in prompt.lower() or "match" in prompt.lower()), \
        "Prompt must handle the no-results case"


# ---------------------------------------------------------------------------
# 6. Anti-hallucination constraint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_has_anti_hallucination_constraint(db, user_model_store):
    """Prompt must forbid inventing dates, names, or content."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "never invent" in prompt.lower() or "do not invent" in prompt.lower() or \
           ("ground" in prompt.lower() and "search results" in prompt.lower()), \
        "Prompt must explicitly forbid hallucination"


# ---------------------------------------------------------------------------
# 7. Prose output (no unnecessary bullets)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_instructs_plain_prose(db, user_model_store):
    """Prompt must instruct plain prose output rather than bullet-point lists."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "prose" in prompt.lower() or "bullet" in prompt.lower(), \
        "Prompt must address output format (prose vs bullets)"


# ---------------------------------------------------------------------------
# 8. Mood-based brevity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_references_mood_for_brevity(db, user_model_store):
    """Prompt must instruct LLM to be brief when mood context shows high stress."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "stress" in prompt.lower() or "mood" in prompt.lower(), \
        "Prompt must reference mood signals for response calibration"
    assert "brief" in prompt.lower() or "concise" in prompt.lower() or "direct" in prompt.lower(), \
        "Prompt must instruct brevity/directness under high stress"


# ---------------------------------------------------------------------------
# 9. search_life() passes the prompt to _query_local
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_life_passes_prompt_to_query_local(db, user_model_store):
    """search_life() must invoke _query_local with the synthesis guide prompt."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_search_context", return_value="ctx") as mock_ctx:
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Here are the results."
            engine.vector_store = None  # disable real vector search
            result = await engine.search_life("what did Alice say?")

    # _query_local must have been called exactly once
    mock_local.assert_called_once()

    system_prompt, user_message = mock_local.call_args[0]

    # System prompt must be the synthesis guide, not the old 3-line stub
    assert len(system_prompt) > 200, \
        "System prompt must be the detailed synthesis guide, not the 3-line stub"
    assert "temporal" in system_prompt.lower() or "citation" in system_prompt.lower(), \
        "System prompt must contain synthesis guide content"


# ---------------------------------------------------------------------------
# 10. Return value forwarded unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_life_returns_model_response_unchanged(db, user_model_store):
    """search_life() must return the model's response string unmodified."""
    engine = AIEngine(db, user_model_store, {})
    expected = "Alice mentioned the Q3 budget on March 5th in a Slack message."

    with patch.object(engine.context, "assemble_search_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = expected
            engine.vector_store = None
            result = await engine.search_life("Q3 budget")

    assert result == expected, "search_life() must return the model response unchanged"


# ---------------------------------------------------------------------------
# 11. Disambiguation section present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_disambiguation_section_present(db, user_model_store):
    """Prompt must have an explicit DISAMBIGUATION section."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "DISAMBIGUATION" in prompt, \
        "Prompt must have a DISAMBIGUATION section header"


# ---------------------------------------------------------------------------
# 12. Thematic grouping of related results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_instructs_thematic_grouping(db, user_model_store):
    """Prompt must tell LLM to group related results thematically."""
    prompt = await _capture_search_prompt(db, user_model_store)
    assert "group" in prompt.lower() or "thematic" in prompt.lower() or "chronologic" in prompt.lower(), \
        "Prompt must instruct grouping of related results rather than arbitrary ordering"
