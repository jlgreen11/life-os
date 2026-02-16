"""
Life OS — AI Engine

Main orchestrator for all LLM interactions. Uses a local model (via Ollama) for
fast, private operations like triage and classification, and optionally
calls a cloud API (Claude) for complex reasoning — with PII stripped.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore
from storage.vector_store import VectorStore
from services.ai_engine.context import ContextAssembler
from services.ai_engine.pii import PIIShield


class AIEngine:
    """
    Main AI orchestrator. Routes queries to the appropriate model,
    manages context, and handles PII protection.
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore, config: dict[str, Any],
                 vector_store: VectorStore = None):
        """
        Initialize the AI Engine.

        Args:
            db: Database manager for event and state storage
            ums: User model store for preferences and behavioral profiles
            config: Configuration dict for model selection and API keys
            vector_store: Optional vector store for semantic search. If not provided,
                         search_life() will fall back to SQL LIKE pattern matching.
        """
        # Core dependencies: database for event/state storage, user model store for
        # preferences and behavioral profiles, config for model selection.
        self.db = db
        self.ums = ums
        self.config = config
        self.vector_store = vector_store
        # ContextAssembler pulls relevant data from DB and user model to build the
        # prompt context window; PIIShield strips sensitive data before cloud calls.
        self.context = ContextAssembler(db, ums)
        self.pii_shield = PIIShield()

        # --- Model configuration ---
        # Local model (Ollama): used for fast, private operations (triage,
        # classification, briefings). All data stays on-device.
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "mistral")
        # Cloud model (Anthropic Claude): used only when higher reasoning quality
        # is needed (e.g., draft replies). Requires an API key AND explicit opt-in
        # via use_cloud=True. PII is always stripped before sending to the cloud.
        self.cloud_api_key = config.get("cloud_api_key")
        self.cloud_model = config.get("cloud_model", "claude-sonnet-4-5-20250514")
        # Cloud usage is gated: both the flag and a valid API key must be present.
        self.use_cloud = config.get("use_cloud", False) and self.cloud_api_key

    async def generate_briefing(self) -> str:
        """Generate the morning briefing."""
        # Step 1: Assemble a rich context window containing calendar events,
        # pending tasks, unread messages, user preferences, mood signals, and
        # known semantic facts. The context assembler handles token budgeting.
        context = self.context.assemble_briefing_context()

        # Step 2: The system prompt instructs the LLM to stay grounded in the
        # provided context -- it must never hallucinate events or tasks.
        # Verbosity is adjusted based on user preferences embedded in the context.
        system_prompt = """You are a private personal assistant. Generate a concise
morning briefing based on the context provided. Be warm but efficient.
Prioritize what needs attention today. If the user prefers minimal verbosity,
keep it very short. Never invent information not in the context."""

        # Step 3: Briefings always use the local model to keep data fully private.
        response = await self._query_local(system_prompt, context)
        return response

    async def draft_reply(self, contact_id: str, channel: str,
                          incoming_message: str) -> str:
        """Draft a reply in the user's voice."""
        # Build context that includes: the user's communication template for this
        # contact/channel (greeting, closing, formality, emoji usage, common
        # phrases), relationship history, and their general linguistic profile.
        context = self.context.assemble_draft_context(
            contact_id, channel, incoming_message
        )

        # The system prompt enforces strict voice-matching: the LLM must produce
        # output indistinguishable from the user's own writing. It outputs only
        # the raw message text -- no preamble, no explanations.
        system_prompt = """You are drafting a reply on behalf of the user.
Match their communication style exactly based on the style profile provided.
Use their typical greeting, closing, formality level, and message length.
The reply should sound indistinguishable from something they'd write themselves.
Do NOT add anything they wouldn't say. Output ONLY the message text."""

        # Cloud path: preferred for draft quality. PII is stripped from both the
        # assembled context and the incoming message before sending to the cloud.
        # Both mappings are merged so that restoration covers all replaced tokens.
        if self.use_cloud:
            stripped_context, mapping = self.pii_shield.strip(context)
            stripped_message, msg_mapping = self.pii_shield.strip(incoming_message)
            # Merge the two PII mappings so restore() can replace all tokens.
            mapping.update(msg_mapping)

            response = await self._query_cloud(system_prompt, stripped_context)
            # Post-process: reinsert real PII values into the cloud's response
            # so the final draft contains the actual names, emails, etc.
            return self.pii_shield.restore(response, mapping)
        else:
            # Fallback: use the local model when cloud is unavailable or disabled.
            # No PII stripping needed since data stays on-device.
            return await self._query_local(system_prompt, context)

    async def extract_action_items(self, text: str, source: str) -> list[dict]:
        """Extract action items from a message or email."""
        # The LLM is instructed to return structured JSON. The prompt constrains
        # the output schema to an array of {title, due_hint, priority} objects.
        system_prompt = """Extract any action items or tasks from the following text.
Return a JSON array of objects with keys: "title", "due_hint" (if any date is
mentioned), "priority" (high/normal/low). If there are no action items, return [].
Return ONLY valid JSON, no other text."""

        # Always uses the local model -- action extraction is fast, private,
        # and does not require cloud-level reasoning.
        response = await self._query_local(system_prompt, text)

        try:
            # Defensive parsing: LLMs sometimes wrap JSON in markdown code fences
            # (```json ... ```). Strip those before parsing.
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove the opening ```json line and the closing ```
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            # Graceful degradation: if the LLM returns unparseable output,
            # return an empty list rather than crashing the pipeline.
            return []

    async def classify_priority(self, event: dict) -> str:
        """Classify the priority of an incoming event."""
        # Extract the key signals for priority classification: sender identity,
        # subject line, and a short preview of the message body.
        payload = event.get("payload", {})
        subject = payload.get("subject", "")
        snippet = payload.get("snippet", "")
        from_addr = payload.get("from_address", "")

        # Truncate the snippet to 200 chars to keep the prompt small and fast.
        # The LLM only needs a preview to judge urgency.
        prompt = f"""Classify the priority of this incoming message.
From: {from_addr}
Subject: {subject}
Preview: {snippet[:200]}

Respond with exactly one word: critical, high, normal, or low."""

        # Uses the local model with a tightly constrained system prompt to
        # force single-word output -- minimizes latency for triage decisions.
        response = await self._query_local(
            "You are a message priority classifier. Respond with one word only.",
            prompt,
        )

        # Validate the LLM's response against the allowed priority levels.
        # If the model returns anything unexpected, default to "normal" to
        # avoid blocking the pipeline with an invalid priority value.
        response_lower = response.strip().lower()
        if response_lower in ["critical", "high", "normal", "low"]:
            return response_lower
        return "normal"

    async def search_life(self, query: str) -> str:
        """
        Search across the user's entire digital life using semantic vector search.

        This method performs intelligent semantic search rather than simple keyword
        matching. It can understand queries like "What did Mike say about the Denver
        project last month?" or "Find that recipe my mom sent me" by matching the
        semantic meaning of the query against embedded event content.

        Args:
            query: Natural language search query

        Returns:
            Natural language answer synthesized from search results
        """
        # Start with a base context describing the user's intent.
        context = self.context.assemble_search_context(query)

        results = []

        # --- Semantic search layer (primary) ---
        # Use vector similarity search if available. This finds semantically related
        # content even when exact keywords don't match. For example, "project update"
        # will match "status report" or "progress check-in".
        if self.vector_store:
            try:
                # Query the vector store for the top 20 most semantically similar
                # documents. The vector store uses cosine similarity on 384-dim
                # embeddings from all-MiniLM-L6-v2 to find relevant content.
                vector_results = self.vector_store.search(query, limit=20)

                # Convert vector store results into the common format expected by
                # the LLM synthesis layer. Vector results include event_id which we
                # use to fetch full event details from the database.
                for vr in vector_results:
                    # Each vector result contains: event_id, text, similarity_score
                    # We fetch the full event from the DB to get type, source, timestamp
                    with self.db.get_connection("events") as conn:
                        row = conn.execute(
                            """SELECT type, source, timestamp, payload FROM events
                               WHERE id = ?""",
                            (vr["event_id"],),
                        ).fetchone()

                    if row:
                        payload = json.loads(row["payload"])
                        results.append({
                            "type": row["type"],
                            "source": row["source"],
                            "date": row["timestamp"],
                            # Include the snippet from the vector store (already
                            # extracted at indexing time) for consistency
                            "snippet": payload.get("snippet", payload.get("subject", ""))[:100],
                            # Include similarity score for debugging/transparency
                            "relevance": round(vr.get("similarity", 0.0), 3),
                        })
            except Exception as e:
                # Graceful degradation: if vector search fails for any reason
                # (database error, model loading issue, etc.), fall back to SQL.
                # This ensures search always returns something even if semantic
                # search is unavailable.
                print(f"Vector search failed, falling back to SQL LIKE: {e}")
                self.vector_store = None  # Disable for this session

        # --- SQL fallback layer ---
        # If vector store is unavailable or disabled, fall back to simple SQL LIKE
        # pattern matching. This is less intelligent but provides a reliable baseline.
        if not self.vector_store or not results:
            with self.db.get_connection("events") as conn:
                rows = conn.execute(
                    """SELECT type, source, timestamp, payload FROM events
                       WHERE payload LIKE ?
                       ORDER BY timestamp DESC LIMIT 20""",
                    (f"%{query}%",),
                ).fetchall()

            # Convert SQL rows into the common result format
            for row in rows:
                payload = json.loads(row["payload"])
                results.append({
                    "type": row["type"],
                    "source": row["source"],
                    "date": row["timestamp"],
                    # Prefer "snippet" field; fall back to "subject" if absent.
                    "snippet": payload.get("snippet", payload.get("subject", ""))[:100],
                })

        # --- Result formatting layer ---
        # Append formatted results to the context for the LLM to synthesize.
        # The LLM receives a JSON array of search hits with type, source, date,
        # snippet, and (optionally) relevance score.
        if results:
            context += f"\n\nSearch results:\n{json.dumps(results, indent=2)}"

        # --- LLM synthesis layer ---
        # The local model synthesizes a natural-language answer from the raw
        # search results. It is instructed to cite specific dates, sources, and
        # people, and to honestly report when no results match the query.
        system_prompt = """You are searching across the user's digital life.
Based on the search results, provide a helpful answer. Be specific about
dates, sources, and people. If no results match, say so honestly."""

        return await self._query_local(system_prompt, context)

    # -------------------------------------------------------------------
    # LLM Query Methods
    # -------------------------------------------------------------------

    async def _query_local(self, system_prompt: str, user_prompt: str) -> str:
        """Query the local Ollama model."""
        # Use httpx async client with a generous 120s timeout -- local models
        # can be slow on CPU-only hardware or with large context windows.
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Ollama exposes an OpenAI-compatible /api/chat endpoint.
            # stream=False requests a single complete response (no SSE chunks).
            response = await client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
            # Raises httpx.HTTPStatusError on 4xx/5xx -- callers should handle.
            response.raise_for_status()
            data = response.json()
            # Extract the assistant's message content; default to empty string
            # if the response structure is unexpected.
            return data.get("message", {}).get("content", "")

    async def _query_cloud(self, system_prompt: str, user_prompt: str) -> str:
        """Query a cloud LLM API (Anthropic Claude) with PII-stripped content."""
        # Safety check: if no API key is configured, fall back to the local model
        # transparently. This makes the cloud path optional without caller changes.
        if not self.cloud_api_key:
            return await self._query_local(system_prompt, user_prompt)

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Call the Anthropic Messages API. The system prompt is passed as a
            # top-level field (not inside messages) per the Anthropic API spec.
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.cloud_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cloud_model,
                    # Cap output at 2048 tokens to control cost and latency.
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            # The Anthropic API returns a list of content blocks. We extract
            # the first text block. Non-text blocks (e.g., tool_use) are ignored.
            content = data.get("content", [])
            if content and content[0].get("type") == "text":
                return content[0]["text"]
            return ""
