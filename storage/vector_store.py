"""
Life OS — Vector Store (LanceDB)

Provides semantic search across the user's entire digital life.
Every piece of content — emails, messages, notes, transactions —
is embedded and stored here for natural language retrieval.

"What did Mike say about the Denver project last month?"
"When did I last discuss solar panels?"
"Find that recipe my mom sent me"

All of these work through vector similarity search.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database for semantic search using LanceDB.
    Falls back to a simple NumPy-based store if LanceDB isn't available.

    Dual-backend strategy:
        Primary  — LanceDB, a columnar vector database with built-in ANN search.
                   Provides production-grade performance and disk-backed persistence.
        Fallback — A NumPy-based in-memory store with JSON persistence.  Used when
                   LanceDB is not installed (e.g. lightweight dev environments).

    The embedding model (``all-MiniLM-L6-v2`` by default) produces 384-dimensional
    normalized vectors, enabling cosine similarity via a simple dot product.
    """

    def __init__(self, db_path: str = "./data/vectors", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        # ``model_name`` refers to a sentence-transformers model identifier.
        # all-MiniLM-L6-v2 is a good balance between speed and quality (384 dims).
        self.model_name = model_name
        self._db = None          # LanceDB connection (or None if fallback)
        self._table = None       # LanceDB table handle for the "documents" table
        self._embedder = None    # SentenceTransformer model instance
        self._use_lancedb = False  # Flag: True if LanceDB initialized successfully

        # Fallback: simple in-memory store backed by a JSON file on disk.
        # Documents and their embedding vectors are kept in parallel lists
        # (index N in _fallback_docs corresponds to index N in _fallback_embeddings).
        self._fallback_docs: list[dict] = []
        self._fallback_embeddings: list[list[float]] = []

    def initialize(self):
        """Initialize the vector store and embedding model.

        Initialization follows a graceful-degradation approach:
        1. Try to import and connect to LanceDB.  If the package is missing,
           fall back to the NumPy-based JSON store.
        2. Try to load the sentence-transformers embedding model.  If the
           package is missing, the store can still accept pre-embedded vectors
           but ``embed_text`` will return None, and searches will fall back
           to simple keyword matching.
        """
        self.db_path.mkdir(parents=True, exist_ok=True)

        # --- Backend selection: prefer LanceDB, fall back to NumPy ---
        try:
            import lancedb
            self._db = lancedb.connect(str(self.db_path / "lance"))
            self._use_lancedb = True
            self._ensure_table()
            logger.info("Vector store: LanceDB initialized")
        except ImportError:
            logger.warning("Vector store: LanceDB not available, using NumPy fallback")
            self._load_fallback()

        # --- Embedding model loading ---
        # sentence-transformers wraps HuggingFace models in a convenient API.
        # The model is loaded once at startup and reused for all embed calls.
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
            logger.info("Embedding model: %s loaded", self.model_name)
        except ImportError:
            logger.warning(
                "Embedding model: sentence-transformers not available — "
                "install with: pip install sentence-transformers"
            )
            self._embedder = None

    def _ensure_table(self):
        """Create the documents table if it doesn't exist.

        The LanceDB table schema mirrors the fallback JSON structure so that
        both backends store identical fields.  The ``vector`` column uses
        ``list_(float32, 384)`` — the 384 dimension matches all-MiniLM-L6-v2.
        If a different embedding model is used, this fixed size must be updated.
        """
        if not self._use_lancedb or not self._db:
            return

        import lancedb
        import pyarrow as pa

        table_name = "documents"
        try:
            # Attempt to open existing table (fast path for restarts).
            self._table = self._db.open_table(table_name)
        except Exception:
            # Table does not exist yet — create with an explicit PyArrow schema.
            schema = pa.schema([
                pa.field("doc_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 384)),  # MiniLM-L6-v2 output dimension
                pa.field("created_at", pa.string()),
            ])
            self._table = self._db.create_table(
                table_name,
                schema=schema,
            )

    def _load_fallback(self):
        """Load the fallback store from disk.

        Reads the JSON file written by ``_save_fallback`` to restore the
        in-memory document list and embedding matrix from a previous session.
        """
        fallback_path = self.db_path / "fallback.json"
        if fallback_path.exists():
            with open(fallback_path) as f:
                data = json.load(f)
                self._fallback_docs = data.get("docs", [])
                self._fallback_embeddings = data.get("embeddings", [])

    def _save_fallback(self):
        """Save the fallback store to disk.

        Writes both the document metadata and raw embedding vectors to a single
        JSON file.  This is simple but not memory-efficient for large corpora;
        for production use, LanceDB should be preferred.
        """
        fallback_path = self.db_path / "fallback.json"
        with open(fallback_path, "w") as f:
            json.dump({
                "docs": self._fallback_docs,
                "embeddings": self._fallback_embeddings,
            }, f)

    def embed_text(self, text: str) -> Optional[list[float]]:
        """Generate an embedding for a text string.

        Returns None if the embedding model is not loaded, allowing callers
        to gracefully degrade to keyword search.  ``normalize_embeddings=True``
        ensures all vectors are unit-length, so cosine similarity can be
        computed as a simple dot product (avoiding an extra normalization step
        at search time).
        """
        if self._embedder is None:
            return None

        try:
            embedding = self._embedder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error("Embedding error: %s", e)
            return None

    def add_document(self, doc_id: str, text: str,
                     metadata: Optional[dict] = None) -> bool:
        """
        Add a document to the vector store.

        Args:
            doc_id: Unique identifier (usually the event ID)
            text: The text content to embed and store
            metadata: Additional metadata (type, source, timestamp, etc.)

        Returns:
            True if at least one chunk was successfully embedded and stored,
            False if the text was too short or no chunks could be embedded
        """
        # Skip very short texts — they produce low-quality embeddings and
        # are unlikely to contain meaningful searchable content.
        if not text or len(text.strip()) < 10:
            return False

        # Split long texts into overlapping chunks (1000 chars, 100 char overlap)
        # so that each chunk fits within the embedding model's effective context
        # window and retrieval can pinpoint the relevant passage.
        chunks = self._chunk_text(text, max_chars=1000, overlap=100)
        chunks_stored = 0

        for i, chunk in enumerate(chunks):
            embedding = self.embed_text(chunk)
            if embedding is None:
                logger.debug("Embedding failed for chunk %d of doc %s", i, doc_id)
                continue

            # If the document was split into multiple chunks, append a numeric
            # suffix to the doc_id so each chunk has a unique identifier while
            # remaining traceable back to the original document.
            chunk_id = f"{doc_id}_{i}" if len(chunks) > 1 else doc_id
            doc = {
                "doc_id": chunk_id,
                "text": chunk,
                "metadata": json.dumps(metadata or {}),
                "vector": embedding,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            if self._use_lancedb and self._table is not None:
                try:
                    import pyarrow as pa
                    self._table.add([doc])
                    chunks_stored += 1
                except Exception as e:
                    logger.error("LanceDB add error: %s", e)
                    return False
            else:
                self._fallback_docs.append({
                    "doc_id": chunk_id,
                    "text": chunk,
                    "metadata": metadata or {},
                    "created_at": doc["created_at"],
                })
                self._fallback_embeddings.append(embedding)
                chunks_stored += 1
                # Persist every 50 documents to balance write frequency against
                # data-loss risk (at most 49 docs lost on a crash).
                if len(self._fallback_docs) % 50 == 0:
                    self._save_fallback()

        if chunks_stored == 0:
            logger.warning("add_document(%s): no chunks embedded — embedding model may be unavailable", doc_id)
            return False
        return True

    def search(self, query: str, limit: int = 10,
               filter_metadata: Optional[dict] = None) -> list[dict]:
        """
        Semantic search across all stored documents.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of matching documents with scores
        """
        # Embed the query text into the same vector space as the stored documents.
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            # If the embedding model is unavailable, fall back to simple keyword
            # matching so the search endpoint still returns *something* useful.
            return self._text_search_fallback(query, limit)

        # Dispatch to the active backend.
        if self._use_lancedb and self._table is not None:
            return self._lancedb_search(query_embedding, limit, filter_metadata)
        else:
            return self._numpy_search(query_embedding, limit, filter_metadata)

    def _lancedb_search(self, query_embedding: list[float], limit: int,
                        filter_metadata: Optional[dict]) -> list[dict]:
        """Search using LanceDB's built-in vector search."""
        try:
            results = (
                self._table
                .search(query_embedding)
                .limit(limit)
                .to_list()
            )

            return [
                {
                    "doc_id": r["doc_id"],
                    "text": r["text"],
                    "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                    "score": float(r.get("_distance", 0)),
                    "created_at": r.get("created_at"),
                }
                for r in results
            ]
        except Exception as e:
            logger.error("LanceDB search error: %s", e)
            return []

    def _numpy_search(self, query_embedding: list[float], limit: int,
                      filter_metadata: Optional[dict]) -> list[dict]:
        """Fallback search using NumPy cosine similarity.

        Because all embeddings are normalized to unit length (see ``embed_text``),
        cosine similarity reduces to a simple dot product:
            cos(a, b) = (a . b) / (||a|| * ||b||)  =>  a . b  when ||a|| = ||b|| = 1

        The 0.1 similarity threshold filters out results that are essentially
        unrelated to the query — without this floor, every search would return
        ``limit`` results even when none are relevant.
        """
        if not self._fallback_embeddings:
            return []

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(self._fallback_embeddings)

        # Dot product = cosine similarity for unit-length vectors.
        similarities = np.dot(doc_vecs, query_vec)

        # Sort indices by descending similarity and take the top ``limit``.
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Similarity threshold: 0.1 filters out near-random matches.
            # Values below this indicate the document has essentially no
            # semantic relationship to the query.
            if score < 0.1:
                continue

            doc = self._fallback_docs[idx]

            # Post-retrieval metadata filter: check that the document's metadata
            # contains all requested key-value pairs.
            if filter_metadata:
                doc_meta = doc.get("metadata", {})
                if not all(doc_meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append({
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": score,
                "created_at": doc.get("created_at"),
            })

        return results

    def _text_search_fallback(self, query: str, limit: int) -> list[dict]:
        """Simple text search when embeddings aren't available.

        Last-resort search: splits the query into whitespace-separated terms and
        scores each document by the fraction of query terms it contains.  This is
        a bag-of-words approach with no semantic understanding, but it ensures the
        search endpoint returns *something* even without ML dependencies.
        """
        query_terms = query.lower().split()
        results = []

        for doc in self._fallback_docs:
            text_lower = doc["text"].lower()
            # Score = fraction of query terms found in the document text.
            matches = sum(1 for term in query_terms if term in text_lower)
            if matches > 0:
                results.append({
                    "doc_id": doc["doc_id"],
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": matches / len(query_terms),
                    "created_at": doc.get("created_at"),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _chunk_text(self, text: str, max_chars: int = 1000,
                    overlap: int = 100) -> list[str]:
        """Split text into overlapping chunks for embedding.

        Chunking strategy:
        - ``max_chars=1000``: keeps each chunk well within the embedding model's
          effective context window.  Shorter chunks also improve retrieval
          precision (the returned text is closer to the actual answer).
        - ``overlap=100``: ensures that sentences straddling a chunk boundary
          appear in at least one chunk in their entirety, preventing information
          loss at boundaries.

        Sentence-boundary detection:
        Before cutting at ``max_chars``, the algorithm tries to find the last
        natural break point (period-space, period-newline, double newline,
        single newline, or space) in the second half of the chunk window.
        Preferring sentence boundaries keeps chunks semantically coherent.
        The separators are tried in priority order — sentence-ending punctuation
        is preferred over mid-sentence whitespace breaks.
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars

            # Try to break at a natural sentence/paragraph boundary rather than
            # cutting mid-word.  Only look in the second half of the chunk
            # (``> max_chars * 0.5``) to avoid producing very short chunks.
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > max_chars * 0.5:
                        end = start + last_sep + len(sep)
                        break

            chunks.append(text[start:end].strip())
            # Slide the window forward, but overlap by ``overlap`` characters
            # so context is shared between adjacent chunks.
            start = end - overlap

        # Filter out any empty strings that could result from stripping whitespace.
        return [c for c in chunks if c]

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        if self._use_lancedb and self._table is not None:
            try:
                count = self._table.count_rows()
                return {"backend": "lancedb", "document_count": count}
            except Exception:
                return {"backend": "lancedb", "document_count": "unknown"}
        else:
            return {
                "backend": "numpy_fallback",
                "document_count": len(self._fallback_docs),
            }

    def delete_document(self, doc_id: str):
        """Delete a document and all its chunks.

        Because a single document may have been split into multiple chunks
        (e.g. ``doc_id``, ``doc_id_0``, ``doc_id_1``, ...), deletion must
        remove both the exact ID and any chunk-suffixed variants.

        In the fallback store, indices are removed in reverse order to avoid
        shifting subsequent indices during iteration.
        """
        if self._use_lancedb and self._table is not None:
            try:
                # LanceDB SQL filter: match exact doc_id OR any chunked variant.
                self._table.delete(f"doc_id = '{doc_id}' OR doc_id LIKE '{doc_id}_%'")
            except Exception as e:
                logger.error("LanceDB delete error: %s", e)
        else:
            # Collect indices of all matching documents (exact + chunk suffixes).
            indices_to_remove = [
                i for i, d in enumerate(self._fallback_docs)
                if d["doc_id"] == doc_id or d["doc_id"].startswith(f"{doc_id}_")
            ]
            # Remove in reverse order so that popping an element does not
            # invalidate the indices of elements yet to be removed.
            for idx in reversed(indices_to_remove):
                self._fallback_docs.pop(idx)
                self._fallback_embeddings.pop(idx)
            self._save_fallback()
