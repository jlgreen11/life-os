"""
Test suite for AIEngine — the main LLM orchestrator.

AIEngine is a critical service handling all LLM interactions, model routing
(local/cloud), PII protection, context assembly, and multiple AI operations
(briefings, drafts, search, triage, task extraction).

Coverage areas:
1. Initialization and configuration
2. Model routing (local vs. cloud)
3. PII protection flow
4. Briefing generation
5. Reply drafting
6. Task extraction with JSON parsing
7. Priority classification
8. Life search
9. Error handling (network failures, invalid responses)
10. Context assembly integration
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from services.ai_engine.engine import AIEngine
from services.ai_engine.context import ContextAssembler
from services.ai_engine.pii import PIIShield


# -------------------------------------------------------------------
# Initialization and Configuration Tests
# -------------------------------------------------------------------


def test_init_minimal_config(db, user_model_store):
    """AIEngine should initialize with minimal config, using defaults."""
    config = {}
    engine = AIEngine(db, user_model_store, config)

    # Verify all dependencies are stored
    assert engine.db is db
    assert engine.ums is user_model_store
    assert engine.config == config

    # Verify default configuration
    assert engine.ollama_url == "http://localhost:11434"
    assert engine.ollama_model == "mistral"
    assert engine.cloud_model == "claude-sonnet-4-5-20250514"
    assert engine.cloud_api_key is None
    assert engine.use_cloud is False  # No API key = no cloud

    # Verify services are initialized
    assert isinstance(engine.context, ContextAssembler)
    assert isinstance(engine.pii_shield, PIIShield)


def test_init_full_config(db, user_model_store):
    """AIEngine should respect all provided config values."""
    config = {
        "ollama_url": "http://custom-ollama:11434",
        "ollama_model": "llama2",
        "cloud_api_key": "sk-ant-test-key",
        "cloud_model": "claude-opus-4-6",
        "use_cloud": True,
    }
    engine = AIEngine(db, user_model_store, config)

    assert engine.ollama_url == "http://custom-ollama:11434"
    assert engine.ollama_model == "llama2"
    assert engine.cloud_api_key == "sk-ant-test-key"
    assert engine.cloud_model == "claude-opus-4-6"
    # use_cloud is the result of config.get("use_cloud", False) and self.cloud_api_key
    # When both are truthy, it stores the API key (which is truthy)
    assert engine.use_cloud == "sk-ant-test-key"


def test_init_cloud_gated_by_api_key(db, user_model_store):
    """Cloud usage requires BOTH use_cloud=True AND a valid API key."""
    # Case 1: Flag enabled but no key
    config = {"use_cloud": True}
    engine = AIEngine(db, user_model_store, config)
    assert not engine.use_cloud  # None AND True = falsy

    # Case 2: Key present but flag disabled
    config = {"cloud_api_key": "sk-ant-test", "use_cloud": False}
    engine = AIEngine(db, user_model_store, config)
    assert not engine.use_cloud  # False AND "sk-ant-test" = falsy

    # Case 3: Both present
    config = {"cloud_api_key": "sk-ant-test", "use_cloud": True}
    engine = AIEngine(db, user_model_store, config)
    assert engine.use_cloud  # True AND "sk-ant-test" = truthy (API key)


# -------------------------------------------------------------------
# Local Model Query Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_local_success(db, user_model_store):
    """_query_local should successfully call Ollama and extract response."""
    engine = AIEngine(db, user_model_store, {})

    mock_response = {
        "message": {
            "role": "assistant",
            "content": "This is the model's response"
        }
    }

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_local("system prompt", "user prompt")

        # Verify API call structure
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["json"]["model"] == "mistral"
        assert call_args[1]["json"]["messages"][0]["role"] == "system"
        assert call_args[1]["json"]["messages"][0]["content"] == "system prompt"
        assert call_args[1]["json"]["messages"][1]["role"] == "user"
        assert call_args[1]["json"]["messages"][1]["content"] == "user prompt"
        assert call_args[1]["json"]["stream"] is False

        # Verify response extraction
        assert result == "This is the model's response"


@pytest.mark.asyncio
async def test_query_local_empty_response(db, user_model_store):
    """_query_local should return empty string if response structure is unexpected."""
    engine = AIEngine(db, user_model_store, {})

    # Case 1: Missing "message" key
    mock_response = {}

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_local("sys", "usr")
        assert result == ""

    # Case 2: Missing "content" in message
    mock_response = {"message": {"role": "assistant"}}

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_local("sys", "usr")
        assert result == ""


@pytest.mark.asyncio
async def test_query_local_custom_url_and_model(db, user_model_store):
    """_query_local should use configured Ollama URL and model."""
    config = {
        "ollama_url": "http://remote-ollama:9999",
        "ollama_model": "custom-model"
    }
    engine = AIEngine(db, user_model_store, config)

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = {"message": {"content": "ok"}}
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        await engine._query_local("sys", "usr")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://remote-ollama:9999/api/chat"
        assert call_args[1]["json"]["model"] == "custom-model"


# -------------------------------------------------------------------
# Cloud Model Query Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_cloud_success(db, user_model_store):
    """_query_cloud should successfully call Anthropic API and extract response."""
    config = {
        "cloud_api_key": "sk-ant-test-key",
        "cloud_model": "claude-opus-4-6",
        "use_cloud": True,
    }
    engine = AIEngine(db, user_model_store, config)

    mock_response = {
        "content": [
            {"type": "text", "text": "Claude's response"}
        ]
    }

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_cloud("system prompt", "user prompt")

        # Verify API call structure
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.anthropic.com/v1/messages"

        headers = call_args[1]["headers"]
        assert headers["x-api-key"] == "sk-ant-test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"

        payload = call_args[1]["json"]
        assert payload["model"] == "claude-opus-4-6"
        assert payload["max_tokens"] == 2048
        assert payload["system"] == "system prompt"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "user prompt"

        # Verify response extraction
        assert result == "Claude's response"


@pytest.mark.asyncio
async def test_query_cloud_no_api_key_falls_back_to_local(db, user_model_store):
    """_query_cloud should fall back to local model if no API key is configured."""
    engine = AIEngine(db, user_model_store, {})  # No cloud_api_key

    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "local response"

        result = await engine._query_cloud("sys", "usr")

        mock_local.assert_called_once_with("sys", "usr")
        assert result == "local response"


@pytest.mark.asyncio
async def test_query_cloud_empty_content(db, user_model_store):
    """_query_cloud should return empty string if content is missing or wrong type."""
    config = {"cloud_api_key": "sk-ant-test", "use_cloud": True}
    engine = AIEngine(db, user_model_store, config)

    # Case 1: Empty content array
    mock_response = {"content": []}

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_cloud("sys", "usr")
        assert result == ""

    # Case 2: Non-text content block
    mock_response = {"content": [{"type": "tool_use", "id": "123"}]}

    with patch("services.ai_engine.engine.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await engine._query_cloud("sys", "usr")
        assert result == ""


# -------------------------------------------------------------------
# Generate Briefing Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_briefing_uses_local_model(db, user_model_store):
    """Briefings should always use the local model for privacy."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, 'assemble_briefing_context') as mock_context:
        mock_context.return_value = "context data"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Good morning! Here's your briefing."

            result = await engine.generate_briefing()

            # Verify context assembly was called
            mock_context.assert_called_once()

            # Verify local model was called with proper prompts
            mock_local.assert_called_once()
            call_args = mock_local.call_args[0]
            assert "personal assistant" in call_args[0].lower()
            # "concise" was in the old 4-line prompt; the new prompt uses "briefing"
            # and a structured synthesis guide — check for an invariant that must
            # always be in any valid briefing system prompt.
            assert "morning briefing" in call_args[0].lower()
            assert call_args[1] == "context data"

            assert result == "Good morning! Here's your briefing."


# -------------------------------------------------------------------
# Draft Reply Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_reply_uses_cloud_when_enabled(db, user_model_store):
    """Draft replies should prefer cloud model for quality when enabled."""
    config = {"cloud_api_key": "sk-ant-test", "use_cloud": True}
    engine = AIEngine(db, user_model_store, config)

    with patch.object(engine.context, 'assemble_draft_context') as mock_context:
        mock_context.return_value = "style: casual"

        with patch.object(engine.pii_shield, 'strip') as mock_strip:
            # PII shield returns stripped text and mapping
            mock_strip.side_effect = [
                ("stripped context", {"[PERSON_1]": "Alice"}),
                ("stripped message", {"[PERSON_2]": "Bob"}),
            ]

            with patch.object(engine.pii_shield, 'restore') as mock_restore:
                mock_restore.return_value = "Hey Alice, thanks for asking!"

                with patch.object(engine, '_query_cloud', new_callable=AsyncMock) as mock_cloud:
                    mock_cloud.return_value = "Hey [PERSON_1], thanks for asking!"

                    result = await engine.draft_reply(
                        contact_id="alice@example.com",
                        channel="email",
                        incoming_message="Hey, can you help?"
                    )

                    # Verify context was assembled
                    mock_context.assert_called_once_with(
                        "alice@example.com", "email", "Hey, can you help?"
                    )

                    # Verify PII stripping was called twice (context + message)
                    assert mock_strip.call_count == 2

                    # Verify cloud model was called
                    mock_cloud.assert_called_once()
                    call_args = mock_cloud.call_args[0]
                    assert "reply" in call_args[0].lower()  # Updated: prompt now says "ghostwriting a reply"
                    assert "stripped context" in call_args[1]

                    # Verify PII restoration with merged mapping
                    mock_restore.assert_called_once()
                    restore_args = mock_restore.call_args
                    assert restore_args[0][0] == "Hey [PERSON_1], thanks for asking!"
                    merged_mapping = restore_args[0][1]
                    assert merged_mapping["[PERSON_1]"] == "Alice"
                    assert merged_mapping["[PERSON_2]"] == "Bob"

                    assert result == "Hey Alice, thanks for asking!"


@pytest.mark.asyncio
async def test_draft_reply_falls_back_to_local_when_cloud_disabled(db, user_model_store):
    """Draft replies should use local model when cloud is disabled."""
    engine = AIEngine(db, user_model_store, {})  # No cloud config

    with patch.object(engine.context, 'assemble_draft_context') as mock_context:
        mock_context.return_value = "style: formal"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Dear colleague, thank you for reaching out."

            result = await engine.draft_reply(
                contact_id="boss@corp.com",
                channel="email",
                incoming_message="Need the report ASAP"
            )

            # Verify local model was called (no PII stripping)
            mock_local.assert_called_once()
            call_args = mock_local.call_args[0]
            assert "reply" in call_args[0].lower()  # Updated: prompt now says "ghostwriting a reply"
            assert call_args[1] == "style: formal"

            assert result == "Dear colleague, thank you for reaching out."


# -------------------------------------------------------------------
# Extract Action Items Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_action_items_parses_json(db, user_model_store):
    """extract_action_items should parse valid JSON from LLM response."""
    engine = AIEngine(db, user_model_store, {})

    mock_llm_response = json.dumps([
        {"title": "Review PR #123", "due_hint": "by Friday", "priority": "high"},
        {"title": "Update documentation", "due_hint": None, "priority": "normal"},
    ])

    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = mock_llm_response

        result = await engine.extract_action_items(
            text="Can you review PR #123 by Friday and update the docs?",
            source="email.received"
        )

        # Verify local model was called
        mock_local.assert_called_once()
        call_args = mock_local.call_args[0]
        assert "action items" in call_args[0].lower()
        assert "json" in call_args[0].lower()

        # Verify JSON was parsed correctly
        assert len(result) == 2
        assert result[0]["title"] == "Review PR #123"
        assert result[0]["due_hint"] == "by Friday"
        assert result[0]["priority"] == "high"
        assert result[1]["title"] == "Update documentation"
        assert result[1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_extract_action_items_strips_markdown_code_fence(db, user_model_store):
    """extract_action_items should handle JSON wrapped in markdown code fences."""
    engine = AIEngine(db, user_model_store, {})

    # LLMs often wrap JSON in ```json ... ``` blocks
    mock_llm_response = """```json
[
  {"title": "Buy groceries", "due_hint": "today", "priority": "normal"}
]
```"""

    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = mock_llm_response

        result = await engine.extract_action_items("Buy groceries today", "message.received")

        # Should successfully parse despite markdown wrapping
        assert len(result) == 1
        assert result[0]["title"] == "Buy groceries"
        assert result[0]["due_hint"] == "today"


@pytest.mark.asyncio
async def test_extract_action_items_returns_empty_list_on_parse_error(db, user_model_store):
    """extract_action_items should gracefully return [] if JSON parsing fails."""
    engine = AIEngine(db, user_model_store, {})

    # Case 1: Invalid JSON
    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "I found no action items in this message."

        result = await engine.extract_action_items("Just saying hi!", "message.received")
        assert result == []

    # Case 2: Malformed JSON
    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "[{broken json"

        result = await engine.extract_action_items("Some text", "email.received")
        assert result == []


# -------------------------------------------------------------------
# Classify Priority Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_priority_returns_valid_level(db, user_model_store):
    """classify_priority should return one of: critical, high, normal, low."""
    engine = AIEngine(db, user_model_store, {})

    event = {
        "payload": {
            "from_address": "ceo@company.com",
            "subject": "URGENT: Server down",
            "snippet": "Production is completely offline. Need immediate fix."
        }
    }

    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "critical"

        result = await engine.classify_priority(event)

        # Verify prompt construction
        mock_local.assert_called_once()
        call_args = mock_local.call_args[0]
        assert "priority classifier" in call_args[0].lower()
        prompt = call_args[1]
        assert "ceo@company.com" in prompt
        assert "URGENT: Server down" in prompt
        assert "Production is completely offline" in prompt

        assert result == "critical"


@pytest.mark.asyncio
async def test_classify_priority_defaults_to_normal_on_invalid_response(db, user_model_store):
    """classify_priority should default to 'normal' if LLM returns invalid value."""
    engine = AIEngine(db, user_model_store, {})

    event = {
        "payload": {
            "from_address": "friend@example.com",
            "subject": "Dinner plans?",
            "snippet": "Want to grab dinner this weekend?"
        }
    }

    # Case 1: LLM returns unexpected value
    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "medium"  # Not in valid set

        result = await engine.classify_priority(event)
        assert result == "normal"

    # Case 2: LLM returns verbose response instead of single word
    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "This message has a high priority because..."

        result = await engine.classify_priority(event)
        assert result == "normal"


@pytest.mark.asyncio
async def test_classify_priority_case_insensitive(db, user_model_store):
    """classify_priority should accept priority levels in any case."""
    engine = AIEngine(db, user_model_store, {})

    event = {"payload": {"from_address": "test@test.com", "subject": "Test", "snippet": ""}}

    for llm_response, expected in [
        ("HIGH", "high"),
        ("Low", "low"),
        ("CrItIcAl", "critical"),
        ("NORMAL", "normal"),
    ]:
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = llm_response
            result = await engine.classify_priority(event)
            assert result == expected


@pytest.mark.asyncio
async def test_classify_priority_truncates_long_snippets(db, user_model_store):
    """classify_priority should truncate snippets to 200 chars to save tokens."""
    engine = AIEngine(db, user_model_store, {})

    long_snippet = "A" * 500  # 500 chars
    event = {
        "payload": {
            "from_address": "test@test.com",
            "subject": "Long message",
            "snippet": long_snippet
        }
    }

    with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
        mock_local.return_value = "normal"

        await engine.classify_priority(event)

        # Verify snippet was truncated in the prompt
        prompt = mock_local.call_args[0][1]
        assert len(long_snippet[:200]) == 200
        assert "A" * 200 in prompt
        assert "A" * 201 not in prompt


# -------------------------------------------------------------------
# Search Life Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_life_finds_results(db, user_model_store, event_store):
    """search_life should query events DB and synthesize results via LLM."""
    # Insert test events
    event_store.store_event({
        "id": "evt-1",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {
            "subject": "Project Alpha Update",
            "snippet": "The Alpha project is on track for Q1 launch."
        },
        "metadata": {}
    })

    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, 'assemble_search_context') as mock_context:
        mock_context.return_value = "User is searching for: Alpha project"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "The Alpha project is on track for Q1 launch according to an email from Feb 10."

            result = await engine.search_life("Alpha project")

            # Verify context was assembled
            mock_context.assert_called_once_with("Alpha project")

            # Verify local model was called with search results
            mock_local.assert_called_once()
            call_args = mock_local.call_args[0]
            assert "searching across" in call_args[0].lower()

            # Verify search results were appended to context
            context_with_results = call_args[1]
            assert "Alpha project" in context_with_results
            assert "Search results:" in context_with_results
            assert "email.received" in context_with_results
            # The search returns snippet, not subject
            assert "The Alpha project is on track" in context_with_results

            assert "Alpha project is on track" in result


@pytest.mark.asyncio
async def test_search_life_no_results(db, user_model_store):
    """search_life should handle case where no events match the query."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, 'assemble_search_context') as mock_context:
        mock_context.return_value = "User is searching for: nonexistent topic"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "I couldn't find any information about that topic."

            result = await engine.search_life("nonexistent topic")

            # Verify local model was still called (with empty results)
            mock_local.assert_called_once()
            context = mock_local.call_args[0][1]
            # No "Search results:" section should be appended when rows is empty
            assert "nonexistent topic" in context

            assert "couldn't find" in result


@pytest.mark.asyncio
async def test_search_life_limits_results_to_20(db, user_model_store, event_store):
    """search_life should cap results at 20 to respect token budget."""
    # Insert 25 matching events
    for i in range(25):
        event_store.store_event({
            "id": f"evt-{i}",
            "type": "message.received",
            "source": "slack",
            "timestamp": f"2026-02-{10+i:02d}T10:00:00Z",
            "priority": "normal",
            "payload": {"snippet": f"Test message {i}"},
            "metadata": {}
        })

    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, 'assemble_search_context') as mock_context:
        mock_context.return_value = "Searching for: Test"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Found many test messages."

            await engine.search_life("Test")

            # Verify context sent to LLM contains exactly 20 results
            context = mock_local.call_args[0][1]
            parsed_results = json.loads(context.split("Search results:\n")[1])
            assert len(parsed_results) == 20


# -------------------------------------------------------------------
# Integration Tests (Context + PII)
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pii_shield_integration_in_draft_reply(db, user_model_store):
    """Verify PII stripping and restoration works end-to-end in draft_reply."""
    config = {"cloud_api_key": "sk-ant-test", "use_cloud": True}
    engine = AIEngine(db, user_model_store, config)

    # Real PIIShield instance (not mocked) with known names
    engine.pii_shield = PIIShield(known_names=["Alice Cooper"])

    with patch.object(engine.context, 'assemble_draft_context') as mock_context:
        # Context contains PII (name and email address)
        mock_context.return_value = "Recipient: Alice Cooper <alice@example.com>\nFormality: casual"

        with patch.object(engine, '_query_cloud', new_callable=AsyncMock) as mock_cloud:
            # Cloud model returns response with PII tokens
            mock_cloud.return_value = "Hey [PERSON_1], got your message at [EMAIL_1]!"

            result = await engine.draft_reply(
                contact_id="alice@example.com",
                channel="email",
                incoming_message="Hey Alice Cooper, can you send me the report?"
            )

            # Verify cloud was called with stripped content
            cloud_call_context = mock_cloud.call_args[0][1]
            assert "[EMAIL_1]" in cloud_call_context or "[PERSON_1]" in cloud_call_context
            assert "alice@example.com" not in cloud_call_context
            assert "Alice Cooper" not in cloud_call_context

            # Verify final result has PII restored
            assert "Alice Cooper" in result
            assert "alice@example.com" in result
            assert "[PERSON_1]" not in result
            assert "[EMAIL_1]" not in result
