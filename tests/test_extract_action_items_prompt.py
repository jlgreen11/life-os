"""
Tests for the enhanced extract_action_items system prompt.

The system prompt was expanded from 10 lines to an 8-section synthesis guide
that gives the LLM precise instructions for:
- Ownership filtering (only user-owned tasks)
- Skip criteria (marketing, FYI, automated notifications)
- Completed-task detection (past-tense reports vs future obligations)
- Priority inference (high/normal/low signals)
- Due-date extraction (preserve original phrasing)
- Title quality (verb-first, ≤10 words)
- Volume constraint (prefer fewer, higher-quality tasks)
- Output format (strict JSON schema with examples)

Coverage:
1. Prompt contains ownership filter instruction
2. Prompt contains skip criteria for marketing/automated content
3. Prompt instructs completed-task detection using tense
4. Prompt gives priority inference rules (high/low signals)
5. Prompt instructs due_hint preservation (not conversion)
6. Prompt requires verb-first, actionable titles
7. Prompt enforces volume constraint (fewer, higher-quality)
8. Prompt specifies exact JSON schema with all required keys
9. Prompt includes anti-hallucination / no-prose-output constraint
10. extract_action_items() passes the enhanced prompt to the local model
11. Return value is parsed JSON list unchanged from LLM response
12. Empty list returned when LLM returns [] for marketing content
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from services.ai_engine.engine import AIEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_system_prompt(db, user_model_store, llm_response: str = "[]") -> str:
    """Run extract_action_items and capture the system prompt passed to _query_local."""
    import asyncio
    engine = AIEngine(db, user_model_store, {})
    captured = {}

    async def _capture(system_prompt, user_prompt):
        captured["prompt"] = system_prompt
        return llm_response

    with patch.object(engine, "_query_local", side_effect=_capture):
        asyncio.get_event_loop().run_until_complete(
            engine.extract_action_items("Please review the proposal by Friday.", "email.received")
        )

    return captured["prompt"]


# ---------------------------------------------------------------------------
# Section 1 — Ownership filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_ownership_filter(db, user_model_store):
    """Prompt must instruct LLM to only extract tasks the user must personally do."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Please schedule a call.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "owner" in prompt_lower or "personally" in prompt_lower or "recipient" in prompt_lower, \
        "Prompt must filter to tasks owned by / assigned to the user personally"


# ---------------------------------------------------------------------------
# Section 2 — Skip criteria
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_skip_criteria_for_marketing(db, user_model_store):
    """Prompt must tell LLM to skip marketing and promotional content."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("50% off today only!", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "marketing" in prompt_lower or "promotional" in prompt_lower or "newsletter" in prompt_lower, \
        "Prompt must explicitly list marketing/promotional as a skip criterion"


@pytest.mark.asyncio
async def test_prompt_contains_skip_criteria_for_automated_notifications(db, user_model_store):
    """Prompt must tell LLM to skip automated system notifications."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Your invoice is ready.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "automated" in prompt_lower or "notification" in prompt_lower or "system" in prompt_lower, \
        "Prompt must list automated system notifications as a skip criterion"


# ---------------------------------------------------------------------------
# Section 3 — Completed task detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_completed_task_detection(db, user_model_store):
    """Prompt must distinguish future obligations from past-tense reports."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("I sent the report yesterday.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "completed" in prompt_lower, \
        "Prompt must reference the 'completed' field for past-tense task detection"
    assert "past" in prompt_lower or "tense" in prompt_lower or "already" in prompt_lower or \
           "future" in prompt_lower, \
        "Prompt must contrast future obligations vs already-completed work"


# ---------------------------------------------------------------------------
# Section 4 — Priority inference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_priority_inference_rules(db, user_model_store):
    """Prompt must give concrete signals for high/normal/low priority assignment."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Urgent: fix the server now.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "urgent" in prompt_lower or "asap" in prompt_lower or "deadline" in prompt_lower, \
        "Prompt must list urgency keywords that elevate to high priority"
    assert "no rush" in prompt_lower or "when you get a chance" in prompt_lower or \
           "eventually" in prompt_lower, \
        "Prompt must list low-priority signals"


# ---------------------------------------------------------------------------
# Section 5 — Due-date extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_due_hint_preservation(db, user_model_store):
    """Prompt must tell LLM to preserve original date phrasing, not convert to absolute dates."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Finish by next Friday.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "due_hint" in prompt_lower or "due hint" in prompt_lower, \
        "Prompt must reference the due_hint field"
    assert "phrasing" in prompt_lower or "original" in prompt_lower or "preserve" in prompt_lower, \
        "Prompt must instruct preservation of original date language"


# ---------------------------------------------------------------------------
# Section 6 — Title quality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_requires_verb_first_actionable_titles(db, user_model_store):
    """Prompt must instruct LLM to write verb-first, actionable task titles."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Review the contract.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "verb" in prompt_lower or "actionable" in prompt_lower, \
        "Prompt must require verb-first, actionable task titles"


# ---------------------------------------------------------------------------
# Section 7 — Volume constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_enforces_volume_constraint(db, user_model_store):
    """Prompt must tell LLM to prefer fewer, higher-quality tasks."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Let me know if you have any questions.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "fewer" in prompt_lower or "quality" in prompt_lower or "duplicate" in prompt_lower, \
        "Prompt must enforce a volume constraint favouring quality over quantity"


# ---------------------------------------------------------------------------
# Section 8 — Output format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_specifies_json_schema_with_all_keys(db, user_model_store):
    """Prompt must list all four required JSON keys in the output schema."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Schedule the demo.", "email.received")

    prompt = mock_local.call_args[0][0]
    # All four output keys must be documented
    assert '"title"' in prompt or "'title'" in prompt or "title" in prompt.lower(), \
        "Prompt must include 'title' in the output schema"
    assert '"due_hint"' in prompt or "due_hint" in prompt, \
        "Prompt must include 'due_hint' in the output schema"
    assert '"priority"' in prompt or "priority" in prompt.lower(), \
        "Prompt must include 'priority' in the output schema"
    assert '"completed"' in prompt or "completed" in prompt.lower(), \
        "Prompt must include 'completed' in the output schema"


@pytest.mark.asyncio
async def test_prompt_contains_no_prose_output_constraint(db, user_model_store):
    """Prompt must explicitly require JSON-only output (no prose, no fences)."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Call the client.", "email.received")

    prompt = mock_local.call_args[0][0]
    prompt_lower = prompt.lower()
    assert "only json" in prompt_lower or "only valid json" in prompt_lower or \
           "no prose" in prompt_lower or "no markdown" in prompt_lower, \
        "Prompt must forbid prose or markdown in the output"


@pytest.mark.asyncio
async def test_prompt_contains_worked_examples(db, user_model_store):
    """Prompt must include at least two concrete input→output examples."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        await engine.extract_action_items("Finish the slides.", "email.received")

    prompt = mock_local.call_args[0][0]
    # Count examples by looking for "Input:" markers (or equivalent patterns)
    example_count = prompt.lower().count("input:")
    assert example_count >= 2, \
        f"Prompt must include ≥2 worked examples; found {example_count}"


# ---------------------------------------------------------------------------
# Integration: model routing and parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_action_items_uses_local_model(db, user_model_store):
    """extract_action_items must always use the local model (privacy requirement)."""
    engine = AIEngine(db, user_model_store, {"cloud_api_key": "sk-test", "use_cloud": True})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        with patch.object(engine, "_query_cloud", new_callable=AsyncMock) as mock_cloud:
            mock_local.return_value = "[]"
            await engine.extract_action_items("Review the deck by Monday.", "email.received")

    mock_cloud.assert_not_called()
    mock_local.assert_called_once()


@pytest.mark.asyncio
async def test_extract_action_items_returns_parsed_list(db, user_model_store):
    """extract_action_items must parse and return the LLM JSON response as a list."""
    engine = AIEngine(db, user_model_store, {})
    expected = [{"title": "Review deck", "due_hint": "Monday", "priority": "high", "completed": False}]

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = json.dumps(expected)
        result = await engine.extract_action_items("Review the deck by Monday.", "email.received")

    assert result == expected


@pytest.mark.asyncio
async def test_extract_action_items_returns_empty_for_marketing(db, user_model_store):
    """extract_action_items must return [] when LLM correctly identifies no action items."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[]"
        result = await engine.extract_action_items("50% off this weekend! Shop now.", "email.received")

    assert result == [], "Marketing email should yield an empty task list"
