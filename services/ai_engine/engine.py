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
from services.ai_engine.context import ContextAssembler
from services.ai_engine.pii import PIIShield


class AIEngine:
    """
    Main AI orchestrator. Routes queries to the appropriate model,
    manages context, and handles PII protection.
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore, config: dict[str, Any]):
        self.db = db
        self.ums = ums
        self.config = config
        self.context = ContextAssembler(db, ums)
        self.pii_shield = PIIShield()

        # Model configuration
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "mistral")
        self.cloud_api_key = config.get("cloud_api_key")
        self.cloud_model = config.get("cloud_model", "claude-sonnet-4-5-20250514")
        self.use_cloud = config.get("use_cloud", False) and self.cloud_api_key

    async def generate_briefing(self) -> str:
        """Generate the morning briefing."""
        context = self.context.assemble_briefing_context()

        system_prompt = """You are a private personal assistant. Generate a concise 
morning briefing based on the context provided. Be warm but efficient. 
Prioritize what needs attention today. If the user prefers minimal verbosity, 
keep it very short. Never invent information not in the context."""

        response = await self._query_local(system_prompt, context)
        return response

    async def draft_reply(self, contact_id: str, channel: str,
                          incoming_message: str) -> str:
        """Draft a reply in the user's voice."""
        context = self.context.assemble_draft_context(
            contact_id, channel, incoming_message
        )

        system_prompt = """You are drafting a reply on behalf of the user.
Match their communication style exactly based on the style profile provided.
Use their typical greeting, closing, formality level, and message length.
The reply should sound indistinguishable from something they'd write themselves.
Do NOT add anything they wouldn't say. Output ONLY the message text."""

        # Use cloud for better quality if available, with PII stripping
        if self.use_cloud:
            stripped_context, mapping = self.pii_shield.strip(context)
            stripped_message, msg_mapping = self.pii_shield.strip(incoming_message)
            mapping.update(msg_mapping)

            response = await self._query_cloud(system_prompt, stripped_context)
            return self.pii_shield.restore(response, mapping)
        else:
            return await self._query_local(system_prompt, context)

    async def extract_action_items(self, text: str, source: str) -> list[dict]:
        """Extract action items from a message or email."""
        system_prompt = """Extract any action items or tasks from the following text.
Return a JSON array of objects with keys: "title", "due_hint" (if any date is 
mentioned), "priority" (high/normal/low). If there are no action items, return [].
Return ONLY valid JSON, no other text."""

        response = await self._query_local(system_prompt, text)

        try:
            # Try to parse JSON from the response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return []

    async def classify_priority(self, event: dict) -> str:
        """Classify the priority of an incoming event."""
        payload = event.get("payload", {})
        subject = payload.get("subject", "")
        snippet = payload.get("snippet", "")
        from_addr = payload.get("from_address", "")

        prompt = f"""Classify the priority of this incoming message.
From: {from_addr}
Subject: {subject}
Preview: {snippet[:200]}

Respond with exactly one word: critical, high, normal, or low."""

        response = await self._query_local(
            "You are a message priority classifier. Respond with one word only.",
            prompt,
        )

        response_lower = response.strip().lower()
        if response_lower in ["critical", "high", "normal", "low"]:
            return response_lower
        return "normal"

    async def search_life(self, query: str) -> str:
        """Search across the user's entire digital life."""
        context = self.context.assemble_search_context(query)

        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                """SELECT type, source, timestamp, payload FROM events
                   WHERE payload LIKE ? 
                   ORDER BY timestamp DESC LIMIT 20""",
                (f"%{query}%",),
            ).fetchall()

        if rows:
            results = []
            for row in rows:
                payload = json.loads(row["payload"])
                results.append({
                    "type": row["type"],
                    "source": row["source"],
                    "date": row["timestamp"],
                    "snippet": payload.get("snippet", payload.get("subject", ""))[:100],
                })
            context += f"\n\nSearch results:\n{json.dumps(results, indent=2)}"

        system_prompt = """You are searching across the user's digital life.
Based on the search results, provide a helpful answer. Be specific about 
dates, sources, and people. If no results match, say so honestly."""

        return await self._query_local(system_prompt, context)

    # -------------------------------------------------------------------
    # LLM Query Methods
    # -------------------------------------------------------------------

    async def _query_local(self, system_prompt: str, user_prompt: str) -> str:
        """Query the local Ollama model."""
        async with httpx.AsyncClient(timeout=120.0) as client:
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
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    async def _query_cloud(self, system_prompt: str, user_prompt: str) -> str:
        """Query a cloud LLM API (Anthropic Claude) with PII-stripped content."""
        if not self.cloud_api_key:
            return await self._query_local(system_prompt, user_prompt)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.cloud_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cloud_model,
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("content", [])
            if content and content[0].get("type") == "text":
                return content[0]["text"]
            return ""
