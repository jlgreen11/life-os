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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


class VectorStore:
    """
    Vector database for semantic search using LanceDB.
    Falls back to a simple NumPy-based store if LanceDB isn't available.
    """

    def __init__(self, db_path: str = "./data/vectors", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.model_name = model_name
        self._db = None
        self._table = None
        self._embedder = None
        self._use_lancedb = False

        # Fallback: simple in-memory store
        self._fallback_docs: list[dict] = []
        self._fallback_embeddings: list[list[float]] = []

    def initialize(self):
        """Initialize the vector store and embedding model."""
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Try LanceDB first
        try:
            import lancedb
            self._db = lancedb.connect(str(self.db_path / "lance"))
            self._use_lancedb = True
            self._ensure_table()
            print("       Vector store: LanceDB initialized")
        except ImportError:
            print("       Vector store: LanceDB not available, using fallback")
            self._load_fallback()

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
            print(f"       Embedding model: {self.model_name} loaded")
        except ImportError:
            print("       Embedding model: sentence-transformers not available")
            print("       Install with: pip install sentence-transformers")
            self._embedder = None

    def _ensure_table(self):
        """Create the documents table if it doesn't exist."""
        if not self._use_lancedb or not self._db:
            return

        import lancedb
        import pyarrow as pa

        table_name = "documents"
        try:
            self._table = self._db.open_table(table_name)
        except Exception:
            # Create with schema
            schema = pa.schema([
                pa.field("doc_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 384)),  # MiniLM dimension
                pa.field("created_at", pa.string()),
            ])
            self._table = self._db.create_table(
                table_name,
                schema=schema,
            )

    def _load_fallback(self):
        """Load the fallback store from disk."""
        fallback_path = self.db_path / "fallback.json"
        if fallback_path.exists():
            with open(fallback_path) as f:
                data = json.load(f)
                self._fallback_docs = data.get("docs", [])
                self._fallback_embeddings = data.get("embeddings", [])

    def _save_fallback(self):
        """Save the fallback store to disk."""
        fallback_path = self.db_path / "fallback.json"
        with open(fallback_path, "w") as f:
            json.dump({
                "docs": self._fallback_docs,
                "embeddings": self._fallback_embeddings,
            }, f)

    def embed_text(self, text: str) -> Optional[list[float]]:
        """Generate an embedding for a text string."""
        if self._embedder is None:
            return None

        try:
            embedding = self._embedder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
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
            True if successfully added
        """
        if not text or len(text.strip()) < 10:
            return False

        # Chunk long text
        chunks = self._chunk_text(text, max_chars=1000, overlap=100)

        for i, chunk in enumerate(chunks):
            embedding = self.embed_text(chunk)
            if embedding is None:
                continue

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
                except Exception as e:
                    print(f"LanceDB add error: {e}")
                    return False
            else:
                self._fallback_docs.append({
                    "doc_id": chunk_id,
                    "text": chunk,
                    "metadata": metadata or {},
                    "created_at": doc["created_at"],
                })
                self._fallback_embeddings.append(embedding)
                # Save periodically
                if len(self._fallback_docs) % 50 == 0:
                    self._save_fallback()

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
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return self._text_search_fallback(query, limit)

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
            print(f"LanceDB search error: {e}")
            return []

    def _numpy_search(self, query_embedding: list[float], limit: int,
                      filter_metadata: Optional[dict]) -> list[dict]:
        """Fallback search using NumPy cosine similarity."""
        if not self._fallback_embeddings:
            return []

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(self._fallback_embeddings)

        # Cosine similarity (vectors are already normalized)
        similarities = np.dot(doc_vecs, query_vec)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.1:  # Skip very low similarity results
                continue

            doc = self._fallback_docs[idx]

            # Apply metadata filter
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
        """Simple text search when embeddings aren't available."""
        query_terms = query.lower().split()
        results = []

        for doc in self._fallback_docs:
            text_lower = doc["text"].lower()
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
        """Split text into overlapping chunks for embedding."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars

            # Try to break at a sentence boundary
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > max_chars * 0.5:
                        end = start + last_sep + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap

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
        """Delete a document and all its chunks."""
        if self._use_lancedb and self._table is not None:
            try:
                self._table.delete(f"doc_id = '{doc_id}' OR doc_id LIKE '{doc_id}_%'")
            except Exception as e:
                print(f"LanceDB delete error: {e}")
        else:
            indices_to_remove = [
                i for i, d in enumerate(self._fallback_docs)
                if d["doc_id"] == doc_id or d["doc_id"].startswith(f"{doc_id}_")
            ]
            for idx in reversed(indices_to_remove):
                self._fallback_docs.pop(idx)
                self._fallback_embeddings.pop(idx)
            self._save_fallback()
