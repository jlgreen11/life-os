"""
Tests for the enhanced morning briefing system prompt.

The system prompt was expanded from 4 generic lines to a 30-line synthesis
guide that instructs the LLM on how to use each of the 12 context sections
assembled by ContextAssembler (mood → tone, completions → wins acknowledgement,
predictions → relationship nudges, episodes → concrete narrative, etc.).

Coverage:
1. Prompt contains mood-based tone calibration instructions
2. Prompt instructs the LLM to acknowledge completed tasks (recent wins)
3. Prompt instructs the LLM to name priority senders by name/subject
4. Prompt instructs the LLM to weave in relationship predictions naturally
5. Prompt instructs the LLM to surface relevant behavioral patterns/routines
6. Prompt instructs use of semantic facts and episodes for personalization
7. Anti-hallucination constraint is present
8. Verbosity constraint is present (minimal → ≤80 words)
9. Prose output constraint (no section headers)
10. generate_briefing() passes the enhanced prompt to the local model
11. Result is returned unchanged from the model
"""

import pytest
from unittest.mock import AsyncMock, patch

from services.ai_engine.engine import AIEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_briefing_system_prompt(db, user_model_store) -> str:
    """Capture the system prompt actually passed to _query_local."""
    engine = AIEngine(db, user_model_store, {})

    captured = {}

    async def _capture(system_prompt, user_message):
        captured["prompt"] = system_prompt
        return "Good morning!"

    import asyncio

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", side_effect=_capture):
            asyncio.get_event_loop().run_until_complete(engine.generate_briefing())

    return captured["prompt"]


# ---------------------------------------------------------------------------
# Tone calibration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_mood_tone_calibration(db, user_model_store):
    """System prompt must instruct LLM to adjust tone based on mood signals."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Good morning!"
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "mood" in prompt.lower(), "Prompt must reference mood signals for tone"
    assert "stress" in prompt.lower() or "valence" in prompt.lower(), \
        "Prompt must mention stress/valence to guide tone adjustment"
    assert "warm" in prompt.lower() or "encouraging" in prompt.lower(), \
        "Prompt must prescribe a warm/encouraging tone for negative mood"


# ---------------------------------------------------------------------------
# Recent wins acknowledgement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_completions_acknowledgement(db, user_model_store):
    """Prompt must tell LLM to acknowledge recently completed tasks."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Great!"
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "completed" in prompt.lower() or "recent wins" in prompt.lower(), \
        "Prompt must reference recently completed tasks section"
    assert "acknowledgement" in prompt.lower() or "acknowledge" in prompt.lower() or "nice work" in prompt.lower(), \
        "Prompt must instruct LLM to acknowledge wins"


# ---------------------------------------------------------------------------
# Priority sender surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_named_priority_sender_surfacing(db, user_model_store):
    """Prompt must tell LLM to name priority senders and their subjects."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Here we go."
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "priority sender" in prompt.lower() or "unread" in prompt.lower(), \
        "Prompt must reference unread/priority sender context section"
    assert "name" in prompt.lower() or "subject" in prompt.lower(), \
        "Prompt must instruct LLM to surface senders by name and subject"


# ---------------------------------------------------------------------------
# Relationship nudges from predictions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_relationship_nudges_from_predictions(db, user_model_store):
    """Prompt must tell LLM to weave relationship predictions naturally."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Done."
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "prediction" in prompt.lower() or "relationship" in prompt.lower(), \
        "Prompt must reference predictions section for relationship nudges"
    assert "opportunity" in prompt.lower() or "reminder" in prompt.lower() or "maintenance" in prompt.lower(), \
        "Prompt must list the relevant prediction types to surface"


# ---------------------------------------------------------------------------
# Behavioral patterns/routines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_behavioral_pattern_surfacing(db, user_model_store):
    """Prompt must tell LLM to surface relevant behavioral patterns/routines."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Noted."
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "insight" in prompt.lower() or "routine" in prompt.lower(), \
        "Prompt must reference insights/routines context sections"
    assert "pattern" in prompt.lower() or "behavioral" in prompt.lower(), \
        "Prompt must mention behavioral patterns"


# ---------------------------------------------------------------------------
# Semantic facts and episode personalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_semantic_facts_and_episode_use(db, user_model_store):
    """Prompt must tell LLM to use semantic facts and episodes for personalization."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Personal!"
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "semantic" in prompt.lower() or "memory" in prompt.lower(), \
        "Prompt must reference semantic memory facts"
    assert "episode" in prompt.lower(), \
        "Prompt must reference recent episodes for concrete narrative"
    assert "personali" in prompt.lower(), \
        "Prompt must use the word personali(ze/zation)"


# ---------------------------------------------------------------------------
# Anti-hallucination constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_anti_hallucination_constraint(db, user_model_store):
    """Prompt must explicitly forbid inventing information."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Safe."
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "never invent" in prompt_lower or "do not invent" in prompt_lower or \
           "ground every" in prompt_lower or "never hallucinate" in prompt_lower, \
        "Prompt must explicitly prohibit inventing information"


# ---------------------------------------------------------------------------
# Verbosity constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_verbosity_constraint(db, user_model_store):
    """Prompt must reference verbosity preference and minimal word limit."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Short."
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "minimal" in prompt.lower(), \
        "Prompt must reference minimal verbosity preference"
    assert "word" in prompt.lower() or "≤" in prompt or "<=" in prompt, \
        "Prompt must give a word-count cap for minimal mode"


# ---------------------------------------------------------------------------
# Prose output constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_prose_output_no_headers(db, user_model_store):
    """Prompt must instruct the LLM to output prose, not headers or labels."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Prose only."
            await engine.generate_briefing()

    prompt = mock_local.call_args[0][0]
    assert "prose" in prompt.lower(), \
        "Prompt must instruct prose output"
    assert "header" in prompt.lower() or "label" in prompt.lower(), \
        "Prompt must explicitly forbid section headers/labels in output"


# ---------------------------------------------------------------------------
# Integration: model routing and return value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_briefing_always_uses_local_model(db, user_model_store):
    """Briefings must always use the local model regardless of cloud config."""
    config = {"cloud_api_key": "sk-ant-test", "use_cloud": True}
    engine = AIEngine(db, user_model_store, config)

    with patch.object(engine.context, "assemble_briefing_context", return_value="rich ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            with patch.object(engine, "_query_cloud", new_callable=AsyncMock) as mock_cloud:
                mock_local.return_value = "Local briefing"
                result = await engine.generate_briefing()

    # Cloud must never be called for briefings (privacy requirement)
    mock_cloud.assert_not_called()
    mock_local.assert_called_once()
    assert result == "Local briefing"


@pytest.mark.asyncio
async def test_generate_briefing_passes_assembled_context_as_user_message(db, user_model_store):
    """generate_briefing() must pass assembled context as the user message."""
    engine = AIEngine(db, user_model_store, {})

    assembled = "Calendar: meeting at 09:00\nTasks: finish report"

    with patch.object(engine.context, "assemble_briefing_context", return_value=assembled):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Briefing text"
            await engine.generate_briefing()

    call_args = mock_local.call_args[0]
    # Second positional argument is the user message (context)
    assert call_args[1] == assembled, \
        "The assembled context must be passed verbatim as the LLM user message"


@pytest.mark.asyncio
async def test_generate_briefing_returns_model_response_unchanged(db, user_model_store):
    """generate_briefing() must return the model response without modification."""
    engine = AIEngine(db, user_model_store, {})
    expected = "Good morning! You have 3 meetings today..."

    with patch.object(engine.context, "assemble_briefing_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = expected
            result = await engine.generate_briefing()

    assert result == expected
