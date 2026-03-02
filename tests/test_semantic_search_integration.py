"""
Test suite for semantic search integration in AIEngine.

This module tests the integration between AIEngine and VectorStore for
intelligent semantic search capabilities. The semantic search feature enables
natural language queries like "What did Mike say about the Denver project?"
to find relevant content based on meaning rather than exact keyword matching.

Coverage areas:
1. Vector store initialization and dependency injection
2. Semantic search with vector similarity
3. Fallback to SQL LIKE when vector store is unavailable
4. Result formatting and synthesis
5. Error handling and graceful degradation
6. Relevance scoring in results
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from services.ai_engine.engine import AIEngine
from storage.vector_store import VectorStore


# -------------------------------------------------------------------
# Initialization Tests
# -------------------------------------------------------------------


def test_init_with_vector_store(db, user_model_store):
    """AIEngine should accept and store a vector_store dependency."""
    mock_vector_store = Mock(spec=VectorStore)
    config = {}

    engine = AIEngine(db, user_model_store, config, vector_store=mock_vector_store)

    # Verify vector store is stored
    assert engine.vector_store is mock_vector_store


def test_init_without_vector_store(db, user_model_store):
    """AIEngine should work without a vector_store (falls back to SQL)."""
    config = {}

    engine = AIEngine(db, user_model_store, config, vector_store=None)

    # Verify vector store is None (SQL fallback will be used)
    assert engine.vector_store is None


# -------------------------------------------------------------------
# Semantic Search Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_life_uses_vector_store(db, user_model_store, event_store):
    """search_life should use vector store for semantic search when available."""
    # Insert test event
    event_store.store_event({
        "id": "evt-semantic-1",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {
            "subject": "Denver Project Status",
            "snippet": "Mike mentioned the Denver project is ahead of schedule."
        },
        "metadata": {}
    })

    # Create mock vector store that returns semantic results
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = [
        {
            "doc_id": "evt-semantic-1",
            "text": "Mike mentioned the Denver project is ahead of schedule.",
            "score": 0.8542
        }
    ]

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context') as mock_context:
        mock_context.return_value = "User is searching for: Mike Denver project"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Mike said the Denver project is ahead of schedule."

            result = await engine.search_life("What did Mike say about Denver?")

            # Verify vector store was called with the query
            mock_vector_store.search.assert_called_once_with(
                "What did Mike say about Denver?",
                limit=20
            )

            # Verify LLM received results with relevance scores
            context = mock_local.call_args[0][1]
            assert "Search results:" in context
            assert "evt-semantic-1" not in context  # Event ID not exposed to LLM
            assert "email.received" in context
            assert "Denver project" in context
            assert "0.854" in context  # Relevance score rounded to 3 decimals

            assert "Denver project is ahead" in result


@pytest.mark.asyncio
async def test_search_life_formats_vector_results_correctly(db, user_model_store, event_store):
    """search_life should format vector results with type, source, date, snippet, relevance."""
    # Insert multiple test events
    event_store.store_event({
        "id": "evt-vec-1",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {
            "subject": "Recipe from Mom",
            "snippet": "Here's the chocolate chip cookie recipe you asked for."
        },
        "metadata": {}
    })

    event_store.store_event({
        "id": "evt-vec-2",
        "type": "message.received",
        "source": "signal",
        "timestamp": "2026-02-11T14:30:00Z",
        "priority": "normal",
        "payload": {
            "snippet": "Mom also sent a brownie recipe last week."
        },
        "metadata": {}
    })

    # Mock vector store returning both results
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = [
        {"doc_id": "evt-vec-1", "text": "chocolate chip cookie recipe", "score": 0.92},
        {"doc_id": "evt-vec-2", "text": "brownie recipe", "score": 0.78},
    ]

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context') as mock_context:
        mock_context.return_value = "Searching for recipe from mom"

        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Found two recipes from mom."

            await engine.search_life("recipe mom sent")

            # Verify results were formatted correctly
            context = mock_local.call_args[0][1]
            results_json = context.split("Search results:\n")[1]
            results = json.loads(results_json)

            # First result
            assert results[0]["type"] == "email.received"
            assert results[0]["source"] == "gmail"
            assert results[0]["date"] == "2026-02-10T10:00:00Z"
            assert "chocolate chip cookie" in results[0]["snippet"]
            assert results[0]["relevance"] == 0.92

            # Second result
            assert results[1]["type"] == "message.received"
            assert results[1]["source"] == "signal"
            assert results[1]["relevance"] == 0.78


@pytest.mark.asyncio
async def test_search_life_handles_snippet_fallback(db, user_model_store, event_store):
    """search_life should use subject as fallback when snippet is missing."""
    event_store.store_event({
        "id": "evt-no-snippet",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": "2026-02-12T09:00:00Z",
        "priority": "normal",
        "payload": {
            "subject": "Team Meeting",
            # No snippet field
        },
        "metadata": {}
    })

    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = [
        {"doc_id": "evt-no-snippet", "text": "Team Meeting", "score": 0.85}
    ]

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Team meeting scheduled"

            await engine.search_life("team meeting")

            # Verify subject was used as snippet fallback
            context = mock_local.call_args[0][1]
            assert "Team Meeting" in context


@pytest.mark.asyncio
async def test_search_life_truncates_snippets_to_100_chars(db, user_model_store, event_store):
    """search_life should truncate snippets to 100 characters max."""
    long_snippet = "A" * 200  # 200 character snippet

    event_store.store_event({
        "id": "evt-long",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {"snippet": long_snippet},
        "metadata": {}
    })

    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = [
        {"doc_id": "evt-long", "text": long_snippet, "score": 0.90}
    ]

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Found result"

            await engine.search_life("test query")

            # Verify snippet was truncated
            context = mock_local.call_args[0][1]
            results = json.loads(context.split("Search results:\n")[1])
            assert len(results[0]["snippet"]) == 100
            assert results[0]["snippet"] == "A" * 100


# -------------------------------------------------------------------
# SQL Fallback Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_life_falls_back_to_sql_when_no_vector_store(db, user_model_store, event_store):
    """search_life should use SQL LIKE search when vector_store is None."""
    event_store.store_event({
        "id": "evt-sql-1",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {
            "subject": "Quarterly Report",
            "snippet": "The Q1 quarterly report shows 25% growth."
        },
        "metadata": {}
    })

    # No vector store provided
    engine = AIEngine(db, user_model_store, {}, vector_store=None)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Q1 report shows growth"

            result = await engine.search_life("quarterly")

            # Verify SQL LIKE matched the keyword
            context = mock_local.call_args[0][1]
            assert "Search results:" in context
            assert "Quarterly Report" in context or "quarterly report" in context
            assert "25% growth" in context


@pytest.mark.asyncio
async def test_search_life_falls_back_to_sql_on_vector_error(db, user_model_store, event_store):
    """search_life should gracefully fall back to SQL if vector search fails."""
    event_store.store_event({
        "id": "evt-fallback",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {"snippet": "Important update about the server migration."},
        "metadata": {}
    })

    # Vector store that raises an exception
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.side_effect = Exception("Vector DB connection failed")

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Server migration update found"

            result = await engine.search_life("server migration")

            # Verify it fell back to SQL LIKE search
            context = mock_local.call_args[0][1]
            assert "Search results:" in context
            assert "server migration" in context or "Server migration" in context

            # Verify vector store was disabled after the error
            assert engine.vector_store is None


@pytest.mark.asyncio
async def test_search_life_falls_back_when_vector_returns_empty(db, user_model_store, event_store):
    """search_life should try SQL fallback when vector search returns no results."""
    # Insert event that matches SQL LIKE but not vector search
    event_store.store_event({
        "id": "evt-sql-only",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {"snippet": "Meeting notes from yesterday's session."},
        "metadata": {}
    })

    # Vector store returns empty results
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = []

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Found meeting notes"

            await engine.search_life("meeting")

            # Verify SQL fallback was used
            context = mock_local.call_args[0][1]
            assert "Search results:" in context
            assert "Meeting notes" in context or "meeting notes" in context


# -------------------------------------------------------------------
# Edge Cases and Error Handling
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_life_handles_missing_event_in_db(db, user_model_store):
    """search_life should skip vector results if the event no longer exists in DB."""
    # Vector store returns an event ID that doesn't exist in the database
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = [
        {"doc_id": "evt-deleted", "text": "deleted content", "score": 0.95}
    ]

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "No results found"

            await engine.search_life("deleted content")

            # Verify no results were added (event not found in DB)
            context = mock_local.call_args[0][1]
            # Context should not contain search results
            assert "Search results:" not in context or "[]" in context


@pytest.mark.asyncio
async def test_search_life_limits_to_20_results(db, user_model_store):
    """search_life should request max 20 results from vector store."""
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = []

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "No results"

            await engine.search_life("test query")

            # Verify limit=20 was passed to vector store
            mock_vector_store.search.assert_called_once_with("test query", limit=20)


@pytest.mark.asyncio
async def test_search_life_handles_malformed_vector_results(db, user_model_store, event_store):
    """search_life should handle vector results missing expected fields gracefully."""
    event_store.store_event({
        "id": "evt-malformed",
        "type": "email.received",
        "source": "gmail",
        "timestamp": "2026-02-10T10:00:00Z",
        "priority": "normal",
        "payload": {"snippet": "Test content"},
        "metadata": {}
    })

    # Vector store returns result with missing score field
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search.return_value = [
        {"doc_id": "evt-malformed", "text": "Test content"}
        # Missing "score" field
    ]

    engine = AIEngine(db, user_model_store, {}, vector_store=mock_vector_store)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "Found test content"

            await engine.search_life("test")

            # Verify it handles missing score gracefully (defaults to 0.0)
            context = mock_local.call_args[0][1]
            results = json.loads(context.split("Search results:\n")[1])
            assert results[0]["relevance"] == 0.0  # Default when score is missing


@pytest.mark.asyncio
async def test_search_life_no_results_returns_empty_synthesis(db, user_model_store):
    """search_life should let LLM synthesize 'no results' message when nothing found."""
    engine = AIEngine(db, user_model_store, {}, vector_store=None)

    with patch.object(engine.context, 'assemble_search_context', return_value=""):
        with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
            mock_local.return_value = "I couldn't find any results matching your query."

            result = await engine.search_life("nonexistent query string xyz123")

            # Verify LLM was asked to synthesize from empty results
            context = mock_local.call_args[0][1]
            # No "Search results:" section when no results
            assert "Search results:" not in context

            assert "couldn't find" in result.lower()


# -------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_life_end_to_end_with_real_vector_store(db, user_model_store, event_store):
    """End-to-end test verifying AIEngine gracefully handles real VectorStore."""
    # Create a real VectorStore instance (may not have embedding model loaded)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(db_path=tmpdir)

        # Insert an event into the database
        event_id = event_store.store_event({
            "id": "evt-e2e",
            "type": "email.received",
            "source": "gmail",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "snippet": "Machine learning models require large datasets for training."
            },
            "metadata": {}
        })

        # Try to add to vector store (may fail if embedding model not available)
        # This is expected in test environments - AIEngine should fall back to SQL
        try:
            vector_store.add_document(
                doc_id=event_id,
                text="Machine learning models require large datasets for training.",
                metadata={"type": "email.received"}
            )
        except Exception:
            pass  # Expected if embedding model not available

        # Create AIEngine with real vector store
        engine = AIEngine(db, user_model_store, {}, vector_store=vector_store)

        with patch.object(engine.context, 'assemble_search_context', return_value=""):
            with patch.object(engine, '_query_local', new_callable=AsyncMock) as mock_local:
                mock_local.return_value = "ML models need big datasets."

                # Query - should work via either vector search OR SQL fallback
                result = await engine.search_life("Machine learning")

                # Verify LLM was called
                mock_local.assert_called_once()
                # Verify search completed successfully
                assert result == "ML models need big datasets."
