"""
Tests for VectorStore ingest metrics (storage/vector_store.py).

Covers the get_ingest_metrics() observability surface and the per-document
attempt/success/failure counters that feed it. The goal is to make silent
embedding failures visible: when Ollama is down, when encode() raises on
oversized text, or when LanceDB writes fail, operators need a counter they
can query without spelunking through logs.

Test Coverage:
    - Initial state: all counters at zero, no last_embed_at
    - Successful add increments attempts + successes, sets last_embed_at
    - Empty-text submissions increment skipped_empty and 'empty_text' reason
    - Encode failures increment attempts + failures and 'model_error' reason
    - Missing embedder model classifies as 'no_model'
    - success_rate computation across mixed outcomes
    - failure_reasons histogram correctly aggregates across events
    - total_rows reflects rows in the fallback document list
    - get_ingest_metrics output is JSON-serialisable
    - The three-event scenario from the task description: success, empty, raises
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from storage.vector_store import VectorStore


@pytest.fixture
def temp_vector_dir(tmp_path):
    """Temporary directory for vector store data (one per test)."""
    return tmp_path / "vectors"


@pytest.fixture
def working_embedder():
    """Mock SentenceTransformer that produces normalised 384-dim vectors.

    Returns deterministic non-zero vectors so similarity-threshold filters
    upstream don't accidentally exclude test documents.
    """
    embedder = Mock()

    def encode(text, normalize_embeddings=False):
        # Deterministic but text-dependent: hash each word into a dimension,
        # mirroring the style used in tests/test_vector_store.py.
        vec = np.zeros(384)
        for word in text.lower().split():
            vec[hash(word) % 384] += 1.0
        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    embedder.encode.side_effect = encode
    return embedder


@pytest.fixture
def store(temp_vector_dir, working_embedder):
    """VectorStore in fallback (NumPy) mode with a working mock embedder."""
    # Patch the LanceDB / fallback loader entry points so construction stays
    # hermetic — no disk reads, no optional dependencies required.
    with patch("storage.vector_store.VectorStore._ensure_table"), \
         patch("storage.vector_store.VectorStore._load_fallback"):
        s = VectorStore(db_path=str(temp_vector_dir))
        s._use_lancedb = False
        s._embedder = working_embedder
        s._fallback_docs = []
        s._fallback_embeddings = []
        s.db_path.mkdir(parents=True, exist_ok=True)
        yield s


def test_initial_counters_are_zero(store):
    """All ingest counters start at zero after construction."""
    metrics = store.get_ingest_metrics()

    assert metrics["embed_attempts"] == 0
    assert metrics["embed_successes"] == 0
    assert metrics["embed_failures"] == 0
    assert metrics["embed_skipped_empty"] == 0
    assert metrics["failure_reasons"] == {}
    assert metrics["success_rate"] == 0.0
    assert metrics["last_embed_at"] is None
    assert metrics["total_rows"] == 0


def test_successful_add_increments_attempts_and_successes(store):
    """A successful add_document call increments attempts + successes and updates last_embed_at."""
    assert store._last_embed_at is None

    result = store.add_document("event1", "A perfectly fine document for the index.")

    assert result is True
    metrics = store.get_ingest_metrics()
    assert metrics["embed_attempts"] == 1
    assert metrics["embed_successes"] == 1
    assert metrics["embed_failures"] == 0
    assert metrics["embed_skipped_empty"] == 0
    assert metrics["last_embed_at"] is not None
    # The timestamp should round-trip through ISO parsing.
    from datetime import datetime
    assert datetime.fromisoformat(metrics["last_embed_at"]) is not None


def test_empty_text_increments_skipped_and_reason_histogram(store):
    """Submitting empty/short text bumps skipped_empty and the 'empty_text' bucket."""
    result = store.add_document("event2", "")

    assert result is False
    metrics = store.get_ingest_metrics()
    # Empty submissions must NOT count as attempts — they were rejected upstream.
    assert metrics["embed_attempts"] == 0
    assert metrics["embed_skipped_empty"] == 1
    assert metrics["failure_reasons"]["empty_text"] == 1


def test_short_text_also_classified_as_empty(store):
    """Text under 10 characters is also routed to 'empty_text', not 'model_error'."""
    result = store.add_document("event3", "tiny")

    assert result is False
    assert store._embed_skipped_empty == 1
    assert store._failure_reasons["empty_text"] == 1


def test_encode_failure_increments_failures_and_model_error(store):
    """When the embedder raises, the document is recorded as a 'model_error' failure."""
    # Monkeypatch the embedder to raise on every call — simulates Ollama down,
    # oversized payload errors, or any transient encode crash.
    store._embedder.encode.side_effect = RuntimeError("encode boom")

    result = store.add_document("event4", "Content that should fail to embed.")

    assert result is False
    metrics = store.get_ingest_metrics()
    # Encode failures DO count as attempts (the empty-text filter passed).
    assert metrics["embed_attempts"] == 1
    assert metrics["embed_failures"] == 1
    assert metrics["embed_successes"] == 0
    assert metrics["failure_reasons"]["model_error"] == 1


def test_missing_embedder_classified_as_no_model(temp_vector_dir):
    """When self._embedder is None, embedding failures classify as 'no_model'."""
    with patch("storage.vector_store.VectorStore._ensure_table"), \
         patch("storage.vector_store.VectorStore._load_fallback"):
        s = VectorStore(db_path=str(temp_vector_dir))
        s._use_lancedb = False
        s._embedder = None
        s._fallback_docs = []
        s._fallback_embeddings = []
        s.db_path.mkdir(parents=True, exist_ok=True)

        result = s.add_document("event5", "Document submitted with no model loaded.")

    assert result is False
    metrics = s.get_ingest_metrics()
    assert metrics["embed_attempts"] == 1
    assert metrics["embed_failures"] == 1
    assert metrics["failure_reasons"]["no_model"] == 1
    # And NOT model_error — the distinction matters for diagnosing whether
    # to start Ollama vs. debug an encode-time crash.
    assert "model_error" not in metrics["failure_reasons"]


def test_three_event_scenario_from_task_description(store):
    """End-to-end: one success, one empty, one encode-failure — covers the headline scenario.

    Mirrors the task description exactly: three events feed in, the metrics
    snapshot reflects the per-event outcomes and the failure-reason histogram.
    """
    # Event 1: succeeds normally.
    assert store.add_document("good", "Solid document with enough content to embed.") is True

    # Event 2: empty text → skipped before any embedding attempt.
    assert store.add_document("empty", "") is False

    # Event 3: encode raises — flip the side_effect just for this event.
    store._embedder.encode.side_effect = RuntimeError("ollama unavailable")
    assert store.add_document("broken", "This one will hit the encode failure path.") is False

    metrics = store.get_ingest_metrics()

    assert metrics["embed_attempts"] == 2          # good + broken (empty doesn't count)
    assert metrics["embed_successes"] == 1         # good
    assert metrics["embed_failures"] == 1          # broken
    assert metrics["embed_skipped_empty"] == 1     # empty

    # Reason histogram captures both rejection types separately.
    assert metrics["failure_reasons"]["empty_text"] == 1
    assert metrics["failure_reasons"]["model_error"] == 1

    # success_rate is over attempted documents only: 1 success / 2 attempts = 0.5
    assert metrics["success_rate"] == 0.5

    # Exactly one row landed in the index.
    assert metrics["total_rows"] == 1

    # last_embed_at points at the one successful event.
    assert metrics["last_embed_at"] is not None


def test_success_rate_all_success(store):
    """success_rate is 1.0 when every attempt succeeds."""
    store.add_document("a", "First document successfully embedded.")
    store.add_document("b", "Second document successfully embedded.")

    metrics = store.get_ingest_metrics()
    assert metrics["embed_attempts"] == 2
    assert metrics["embed_successes"] == 2
    assert metrics["success_rate"] == 1.0


def test_success_rate_zero_attempts_is_zero(store):
    """success_rate is 0.0 (not NaN) when there have been no attempts."""
    # Only empty submissions — no attempts recorded.
    store.add_document("e1", "")
    store.add_document("e2", "")

    metrics = store.get_ingest_metrics()
    assert metrics["embed_attempts"] == 0
    assert metrics["success_rate"] == 0.0


def test_failure_reasons_aggregates_across_events(store):
    """failure_reasons accumulates counts across multiple failed events."""
    # Two empty submissions
    store.add_document("e1", "")
    store.add_document("e2", "   ")

    # Two encode failures
    store._embedder.encode.side_effect = RuntimeError("crash")
    store.add_document("f1", "Long enough content to pass the empty filter, but encode crashes.")
    store.add_document("f2", "Another long enough doc that will hit the same encode crash path.")

    metrics = store.get_ingest_metrics()
    assert metrics["failure_reasons"]["empty_text"] == 2
    assert metrics["failure_reasons"]["model_error"] == 2


def test_metrics_dict_is_json_serialisable(store):
    """The full metrics snapshot must serialise to JSON for transport over /api."""
    store.add_document("ok", "Decent document to embed.")
    store.add_document("empty", "")
    store._embedder.encode.side_effect = RuntimeError("boom")
    store.add_document("bad", "Document that will fail to encode.")

    metrics = store.get_ingest_metrics()

    # json.dumps would raise if defaultdict, datetime, or numpy scalars leaked through.
    blob = json.dumps(metrics)
    assert json.loads(blob) == metrics


def test_total_rows_reflects_chunks_not_documents(store):
    """total_rows counts stored rows, including chunked variants of long docs."""
    # Short doc → 1 row
    store.add_document("short", "Short doc content.")
    # Long doc → multiple chunk rows
    store.add_document("long", "a" * 2500)

    metrics = store.get_ingest_metrics()
    # 1 (short) + at least 2 (chunked long) = >= 3 rows
    assert metrics["total_rows"] >= 3
    # But only 2 successful events.
    assert metrics["embed_successes"] == 2


def test_total_rows_lancedb_count_failure_returns_unknown(temp_vector_dir):
    """If the LanceDB count_rows query raises, total_rows degrades to 'unknown'."""
    with patch("storage.vector_store.VectorStore._ensure_table"), \
         patch("storage.vector_store.VectorStore._load_fallback"):
        s = VectorStore(db_path=str(temp_vector_dir))
        s._use_lancedb = True
        s._table = Mock()
        s._table.count_rows.side_effect = Exception("table corrupted")

        metrics = s.get_ingest_metrics()

    assert metrics["total_rows"] == "unknown"


def test_get_health_and_get_ingest_metrics_are_independent(store):
    """get_health and get_ingest_metrics expose different surfaces; neither should mutate the other."""
    store.add_document("a", "Indexed content for the diagnostics test.")

    health = store.get_health()
    metrics = store.get_ingest_metrics()

    # Health should not surface the ingest counters (they belong to the
    # separate ingest-metrics surface).
    assert "embed_attempts" not in health
    assert "failure_reasons" not in health

    # And ingest metrics should not include backend-health fields.
    assert "backend" not in metrics
    assert "is_healthy" not in metrics
