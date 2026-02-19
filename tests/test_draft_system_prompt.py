"""
Tests for the enhanced draft_reply() system prompt.

The system prompt was expanded from 5 generic lines to a multi-section
synthesis guide that instructs the LLM on how to use each of the 5 context
layers assembled by ContextAssembler.assemble_draft_context():

  Layer 1 — Communication template (greeting/closing/formality/length)
  Layer 2/3 — Per-contact or global outbound linguistic style metrics
  Layer 4 — Contact's inbound writing style (register calibration)
  Layer 5 — Recent conversation history (thread continuation)

Without this guide the LLM tends to ignore the per-contact data and produce
generic-sounding drafts that don't leverage the 700K+ signal samples collected
by the system.

Coverage:
 1. Prompt instructs use of communication template as highest priority
 2. Prompt instructs verbatim greeting/closing from template
 3. Prompt contains per-contact outbound style interpretation
 4. Prompt contains formality scale definition (0–1 numeric interpretation)
 5. Prompt instructs contact register mirroring from inbound style
 6. Prompt instructs natural (not forced) use of conversation history
 7. Prompt contains relationship depth / interaction-count calibration
 8. Prompt contains anti-hallucination constraint
 9. Prompt enforces output-only constraint (no preamble/meta-commentary)
10. Prompt instructs answering questions directly first
11. Prompt references typical_length target for message length
12. draft_reply() passes the enhanced prompt to the query layer (local path)
13. draft_reply() passes the enhanced prompt to the query layer (cloud path)
14. Return value from model is forwarded unchanged (local path)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.ai_engine.engine import AIEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_draft_prompt(db, user_model_store, *, use_cloud: bool = False) -> str:
    """
    Invoke draft_reply() with a mocked context and query layer, and return
    the system prompt that was actually passed to the model.

    Args:
        db: DatabaseManager fixture (provides real SQLite connections).
        user_model_store: UserModelStore fixture.
        use_cloud: If True, exercise the cloud path; otherwise local path.

    Returns:
        The system prompt string captured from the call to _query_local or
        _query_cloud.
    """
    config = {}
    if use_cloud:
        # Provide fake credentials so use_cloud becomes True.
        config = {"use_cloud": True, "cloud_api_key": "fake-key"}

    engine = AIEngine(db, user_model_store, config)

    captured: dict = {}

    async def _capture_local(system_prompt, user_message):
        captured["prompt"] = system_prompt
        return "Hi there!"

    async def _capture_cloud(system_prompt, user_message):
        captured["prompt"] = system_prompt
        return "Hi there!"

    with patch.object(
        engine.context, "assemble_draft_context", return_value="<stub context>"
    ):
        if use_cloud:
            with patch.object(engine, "_query_cloud", side_effect=_capture_cloud):
                with patch.object(engine.pii_shield, "strip", return_value=("<stripped>", {})):
                    with patch.object(engine.pii_shield, "restore", return_value="Hi there!"):
                        asyncio.get_event_loop().run_until_complete(
                            engine.draft_reply("contact@example.com", "email", "Hello!")
                        )
        else:
            with patch.object(engine, "_query_local", side_effect=_capture_local):
                asyncio.get_event_loop().run_until_complete(
                    engine.draft_reply("contact@example.com", "email", "Hello!")
                )

    return captured.get("prompt", "")


# ---------------------------------------------------------------------------
# Priority 1 — Communication template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_prioritises_communication_template(db, user_model_store):
    """Prompt must declare the communication template as the highest-priority source."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "priority 1" in lower or "highest priority" in lower, (
        "Prompt must declare communication template as priority 1 / highest priority"
    )
    assert "template" in lower, "Prompt must reference communication template"


@pytest.mark.asyncio
async def test_prompt_instructs_verbatim_greeting_closing(db, user_model_store):
    """Prompt must tell LLM to use the template's greeting and closing verbatim."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "greeting" in lower, "Prompt must instruct use of the template greeting"
    assert "closing" in lower, "Prompt must instruct use of the template closing"
    assert "verbatim" in lower, "Prompt must say to use greeting/closing verbatim"


# ---------------------------------------------------------------------------
# Priority 2/3 — Outbound style metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_formality_scale_definition(db, user_model_store):
    """Prompt must explain the 0–1 formality scale so the LLM can apply it correctly."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    # The scale must tell the LLM what 0 and 1 mean.
    assert "0.0" in prompt or "0–" in prompt or "0.3" in prompt, (
        "Prompt must define the lower end of the formality scale"
    )
    assert "formal" in lower, "Prompt must use the word 'formal'"
    assert "casual" in lower, "Prompt must use the word 'casual'"


@pytest.mark.asyncio
async def test_prompt_mentions_per_contact_style(db, user_model_store):
    """Prompt must explain the per-contact style is preferred over global averages."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "per-contact" in lower or "contact" in lower, (
        "Prompt must reference per-contact outbound style data"
    )
    assert "global" in lower, "Prompt must mention global style as the fallback"


@pytest.mark.asyncio
async def test_prompt_guides_hedge_rate_and_question_rate(db, user_model_store):
    """Prompt must explain how to apply question_rate and hedge_rate metrics."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "question_rate" in lower or "question rate" in lower, (
        "Prompt must guide use of question_rate metric"
    )
    assert "hedge" in lower, (
        "Prompt must guide use of hedge_rate metric to soften statements"
    )


# ---------------------------------------------------------------------------
# Priority 4 — Contact register (inbound style)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_contact_register_mirroring(db, user_model_store):
    """Prompt must instruct the LLM to consider the contact's inbound writing register."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "inbound" in lower or "contact's" in lower or "register" in lower, (
        "Prompt must reference the contact's inbound writing style / register"
    )


# ---------------------------------------------------------------------------
# Priority 5 — Conversation history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_natural_history_reference(db, user_model_store):
    """Prompt must tell LLM to reference conversation history naturally, not force it."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "history" in lower or "conversation" in lower, (
        "Prompt must reference the conversation history context layer"
    )
    # Must say something like "when it fits naturally" / "skip if" / "don't force"
    assert (
        "natural" in lower
        or "sparingly" in lower
        or "skip" in lower
        or "force" in lower
    ), "Prompt must guide LLM to use history naturally and not force references"


# ---------------------------------------------------------------------------
# Relationship depth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_relationship_depth_guidance(db, user_model_store):
    """Prompt must explain how interaction count calibrates tone warmth."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "interaction" in lower or "relationship" in lower, (
        "Prompt must reference interaction count / relationship depth"
    )


# ---------------------------------------------------------------------------
# Anti-hallucination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_anti_hallucination_constraint(db, user_model_store):
    """Prompt must forbid inventing facts not present in the context."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "invent" in lower or "never invent" in lower or "ground" in lower, (
        "Prompt must contain an anti-hallucination constraint"
    )


# ---------------------------------------------------------------------------
# Output-only constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_enforces_output_only(db, user_model_store):
    """Prompt must instruct LLM to output only the message text with no preamble."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "output only" in lower or "only the message" in lower, (
        "Prompt must enforce output-only (no preamble or meta-commentary)"
    )
    assert "preamble" in lower or "meta" in lower or "label" in lower, (
        "Prompt must explicitly forbid preamble / labels / meta-commentary"
    )


# ---------------------------------------------------------------------------
# Direct-answer constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_instructs_direct_answer_first(db, user_model_store):
    """Prompt must tell LLM to answer questions directly before adding context."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    lower = prompt.lower()
    assert "question" in lower and "direct" in lower, (
        "Prompt must instruct direct answer to questions first"
    )


# ---------------------------------------------------------------------------
# Message length guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_references_typical_length(db, user_model_store):
    """Prompt must instruct LLM to respect the typical_length target from context."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "reply"
            await engine.draft_reply("a@b.com", "email", "msg")

    prompt = mock_local.call_args[0][0]
    assert "typical_length" in prompt or "word" in prompt.lower(), (
        "Prompt must reference typical_length target for message length guidance"
    )


# ---------------------------------------------------------------------------
# Integration: local path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_reply_passes_prompt_to_local_model(db, user_model_store):
    """draft_reply() must pass the synthesis-guide prompt to _query_local on the local path."""
    engine = AIEngine(db, user_model_store, {})

    with patch.object(engine.context, "assemble_draft_context", return_value="context-stub"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Hello!"
            result = await engine.draft_reply("a@b.com", "email", "incoming msg")

    assert mock_local.call_count == 1, "_query_local must be called exactly once"
    prompt_arg = mock_local.call_args[0][0]
    # The prompt must be the synthesis guide (substantial, not just 5 lines).
    assert len(prompt_arg) > 300, (
        "Enhanced system prompt must be substantial (>300 chars); got a stub"
    )
    assert "PRIORITY 1" in prompt_arg, (
        "Enhanced prompt must contain PRIORITY 1 section marker"
    )


@pytest.mark.asyncio
async def test_draft_reply_returns_model_output_unchanged(db, user_model_store):
    """draft_reply() must forward the model response unchanged on the local path."""
    engine = AIEngine(db, user_model_store, {})
    expected = "Thanks for reaching out!"

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = expected
            result = await engine.draft_reply("a@b.com", "email", "Hello!")

    assert result == expected, "draft_reply must return model output unchanged"


# ---------------------------------------------------------------------------
# Integration: cloud path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_reply_passes_prompt_to_cloud_model(db, user_model_store):
    """draft_reply() must pass the synthesis-guide prompt to _query_cloud on the cloud path."""
    config = {"use_cloud": True, "cloud_api_key": "fake-key-for-test"}
    engine = AIEngine(db, user_model_store, config)

    with patch.object(engine.context, "assemble_draft_context", return_value="ctx"):
        with patch.object(engine, "_query_cloud", new_callable=AsyncMock) as mock_cloud:
            mock_cloud.return_value = "Hello!"
            with patch.object(engine.pii_shield, "strip", return_value=("<stripped>", {})):
                with patch.object(engine.pii_shield, "restore", return_value="Hello!"):
                    await engine.draft_reply("a@b.com", "email", "incoming msg")

    assert mock_cloud.call_count == 1, "_query_cloud must be called exactly once"
    prompt_arg = mock_cloud.call_args[0][0]
    assert len(prompt_arg) > 300, (
        "Enhanced system prompt must be passed to cloud model (>300 chars)"
    )
    assert "PRIORITY 1" in prompt_arg, (
        "Enhanced prompt must contain PRIORITY 1 section marker when using cloud"
    )
