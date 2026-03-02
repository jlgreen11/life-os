"""
Life OS — AI Engine

Main orchestrator for all LLM interactions. Uses a local model (via Ollama) for
fast, private operations like triage and classification, and optionally
calls a cloud API (Claude) for complex reasoning — with PII stripped.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

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

        # Step 2: The system prompt is a synthesis guide for the 12-section
        # context window assembled above.  It tells the LLM *how* to use each
        # section rather than leaving it to guess — mood for tone calibration,
        # completions for acknowledgement, predictions for relationship nudges,
        # episodes for concrete narrative, etc.  Without this guide, the LLM
        # tends to emit a generic task list and ignores the richer signals.
        system_prompt = """You are a private personal assistant generating a personalized morning briefing.

TONE CALIBRATION — check the "User mood context" section:
- If emotional_valence is "negative" or stress_level is "high": be warm, encouraging, compassionate.
- If energy_level is "high" and emotional_valence is "positive": be upbeat and forward-looking.
- If trend is "declining": add one brief supportive note; if "improving": acknowledge the upturn.
- Default to warm but businesslike when mood data is absent.

SYNTHESIS GUIDE — use each context section as follows:
1. Recent wins: If "Recently completed" tasks exist, open with a brief one-line acknowledgement ("You finished X yesterday — nice work."). Skip if empty.
2. Today's focus: Synthesize "Calendar events" + "Pending tasks" into 2-3 concrete action priorities. Name specific events and tasks; do not just say "you have meetings".
3. Inbox pulse: If "Unread messages" lists named priority senders, surface them by name and subject ("Alice sent 2 emails about the Q3 review"). Skip generic unread counts unless nothing more specific is available.
4. Relationship nudges: Weave in any "Active predictions" of type opportunity, reminder, or relationship_maintenance as natural sentences ("You haven't replied to Bob in 12 days — worth a quick note?"). Do not emit these as a separate section.
5. Behavioral patterns: If "Behavioral insights" or "Behavioral routines" include something relevant to today (e.g., a Friday pattern on a Friday, or a cadence observation tied to current workload), mention one briefly. Skip if none apply.
6. Tone from memory: Use "Semantic memory facts" and "Recent episodes" to personalize phrasing — e.g., reference a known preference or ongoing topic — but do not list facts verbatim.

CONSTRAINTS:
- Ground every statement in the provided context. Never invent tasks, names, dates, or events.
- If verbosity preference is "minimal", collapse the entire briefing to ≤80 words and omit section 5.
- Output plain prose paragraphs. Use bullet points only when summarising 4 or more distinct items.
- Do not include section headers or labels in your output."""

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

        # The system prompt is a synthesis guide for the 5-layer context window
        # assembled above.  It tells the LLM *how* to interpret each section —
        # which data takes priority, how to read numeric style metrics, when to
        # reference conversation history, and how to calibrate against the
        # contact's own writing register.  Without this guide the LLM tends to
        # ignore the per-contact data and produce generic-sounding drafts.
        system_prompt = """You are ghostwriting a reply on behalf of the user. Use the context layers below in priority order.

PRIORITY 1 — Communication template (highest priority):
If a "Communication style for this context:" section appears, it was built from real examples with this contact on this channel. Use its greeting and closing verbatim; match its formality label exactly; stay within ±20% of its typical word count; include emoji only if "Uses emoji: yes".

PRIORITY 2 — Per-contact outbound style:
If "User's style with this contact (N msgs)" appears, apply its numeric metrics. Formality scale: 0.0–0.3 = very casual, 0.3–0.6 = neutral, 0.6–1.0 = formal. If question_rate > 0.15 the user asks frequent questions — mirror that. If hedge_rate > 0.10 soften statements ("I think…", "maybe…"). Match avg_sentence_length closely.

PRIORITY 3 — Global style (fallback):
If only "User's general style" appears (no per-contact data), apply the same metric interpretation but recognise it averages across all contacts — err toward the contact's inbound register.

CONTACT REGISTER — Contact's inbound style:
If "Contact's writing style" appears, their formality and sentence length suggest what register feels natural to them. When their formality differs from the user's typical style by >0.15, meet them roughly halfway — don't be stiff with a casual contact or overly breezy with a formal one.

CONVERSATION HISTORY — use sparingly:
If "Recent conversation history" appears, you may acknowledge an ongoing thread when it fits naturally ("Following up on our Q3 discussion…"). Skip if the incoming message opens an unrelated topic; never force a reference.

RELATIONSHIP DEPTH:
The interaction count calibrates warmth. 100+ interactions → comfortable, direct; 10–100 → familiar but considered; <10 → polite and somewhat measured.

CONSTRAINTS:
- Output ONLY the message text. No preamble, no labels, no meta-commentary.
- Ground every sentence in the provided context. Never invent facts, dates, or events.
- If the incoming message is a question, answer it directly first, then add any context.
- Respect the typical_length target: a 20-word template implies a short reply; a 100-word template implies a longer one."""

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
        """Extract action items from a message or email.

        Returns:
            List of action items with keys:
                - title: Task description
                - due_hint: Any mentioned deadline (optional)
                - priority: high/normal/low
                - completed: true if the task is already done, false otherwise

        The AI analyzes whether each action is a future task, a completed action,
        or a notification about someone else's work. This enables immediate workflow
        detection from historical data instead of waiting days for task aging.

        Examples:
            "Please send the report by Friday" → completed: false
            "I sent the report yesterday" → completed: true
            "The report was sent by the team" → completed: true (or skip entirely)
        """
        # The LLM is instructed to return structured JSON. The prompt constrains
        # the output schema to an array of {title, due_hint, priority, completed} objects.
        # The "completed" field is critical for workflow detection — it lets us generate
        # task.completed events immediately from historical emails instead of waiting
        # 7+ days for inactivity detection.
        #
        # SECTION STRUCTURE (8 sections):
        # 1. Role — precision-focused task extraction specialist
        # 2. Ownership filter — only tasks the recipient must personally do
        # 3. Skip criteria — what to exclude (marketing, FYI, completed-by-others)
        # 4. Completed task detection — past-tense reports vs future obligations
        # 5. Priority inference — signals that elevate to high/lower to low
        # 6. Due-date extraction — how to normalise relative dates
        # 7. Title quality — concise, verb-first, actionable
        # 8. Output format + examples + volume constraint
        system_prompt = """You are a task extraction specialist. Your job is to identify \
genuine action items that the message recipient (the user) must personally complete.

## Section 1 — Ownership filter
Only extract tasks where the user is the expected actor. Skip tasks:
- Assigned to or completed by someone else ("The team will...", "Alice sent...")
- Pure announcements with no required response ("Your order has shipped.")
- FYI forwards that require only awareness, not action

## Section 2 — Skip criteria (return [] for these)
- Marketing, promotional, or newsletter content
- Automated system notifications (invoices generated, alerts triggered, cron jobs)
- Social media digests or engagement bait ("You have 5 new followers")
- Calendar invites that merely inform (no RSVP action needed)
- Messages that are entirely conversational ("Thanks!", "Sounds good.")

## Section 3 — Completed task detection
Distinguish future obligations from already-done work:
- Future task (completed: false): imperative verbs, requests, "please", "can you", "by [date]"
- Already done (completed: true): past tense report from the user ("I submitted...", "We shipped...")
- Third-party completion (completed: true OR skip): someone else completed the work

## Section 4 — Priority inference
Assign priority based on urgency and sender weight:
- high: explicit deadlines ("by EOD", "by Friday"), words like "urgent", "ASAP", \
  "critical", "blocking", legal/financial consequences, requests from known bosses/clients
- low: "when you get a chance", "no rush", "eventually", personal errands
- normal: everything else

## Section 5 — Due-date extraction (due_hint field)
Preserve the original phrasing — do not convert to absolute dates:
- "by Friday" → "by Friday"
- "EOD today" → "EOD today"
- "next week" → "next week"
- Omit due_hint (null) when no deadline is stated or implied

## Section 6 — Title quality
Write task titles that are:
- Verb-first and actionable: "Review proposal", "Send invoice to Acme", "Schedule dentist appointment"
- Specific enough to act on without re-reading the source message
- ≤10 words — no filler words like "Please" or "Could you"

## Section 7 — Volume constraint
Prefer fewer, higher-quality tasks. Do NOT extract:
- Duplicate tasks for the same underlying action
- Overly granular sub-steps when a single task covers them
- Speculative tasks ("maybe consider looking into...")
If the message contains no genuine user-owned action items, return [].

## Section 8 — Output format
Return ONLY a valid JSON array. Each element must have exactly these keys:
  "title"    (string, required)
  "due_hint" (string or null)
  "priority" ("high" | "normal" | "low", required)
  "completed" (true | false, required)

Return [] if there are no action items. Return ONLY JSON — no prose, no markdown fences.

### Examples
Input: "Can you review the attached proposal by EOD Friday? Alice and I need your sign-off."
Output: [{"title": "Review proposal and provide sign-off", "due_hint": "EOD Friday", "priority": "high", "completed": false}]

Input: "I sent the report to the client yesterday. Just letting you know."
Output: [{"title": "Send report to client", "due_hint": null, "priority": "normal", "completed": true}]

Input: "50% off everything this weekend only! Shop now."
Output: []

Input: "Your AWS bill for December is ready. Total: $142.38."
Output: []

Input: "Please review PR #88 when you get a chance, no rush."
Output: [{"title": "Review PR #88", "due_hint": null, "priority": "low", "completed": false}]"""

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
                # the LLM synthesis layer. Vector results include doc_id which we
                # map back to event IDs to fetch full event details from the database.
                #
                # Batched IN query: fetch all matching event rows in a single
                # round-trip rather than issuing one SELECT per result (N+1).
                # This reduces SQLite overhead from O(N) queries to O(1).
                if vector_results:
                    # Build a lookup from event_id → best similarity score.
                    # The vector store returns 'doc_id' and 'score' fields.
                    # When documents are chunked, doc_ids have '_N' suffixes
                    # (e.g. 'evt123_0', 'evt123_1'). Strip these to recover
                    # the original event ID, keeping the best score per event.
                    similarity_by_id: dict[str, float] = {}
                    for vr in vector_results:
                        doc_id = vr["doc_id"]
                        score = vr.get("score", 0.0)
                        # Strip chunk suffix: 'evt_0' → 'evt', but 'evt' stays 'evt'
                        parts = doc_id.rsplit("_", 1)
                        event_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else doc_id
                        # Keep the highest score when multiple chunks match
                        if event_id not in similarity_by_id or score > similarity_by_id[event_id]:
                            similarity_by_id[event_id] = score
                    event_ids = list(similarity_by_id.keys())
                    placeholders = ",".join("?" * len(event_ids))

                    with self.db.get_connection("events") as conn:
                        rows = conn.execute(
                            f"""SELECT id, type, source, timestamp, payload
                                FROM events
                                WHERE id IN ({placeholders})""",
                            event_ids,
                        ).fetchall()

                    # Re-order rows to match the vector store's similarity ranking
                    # (the DB may return them in arbitrary insertion order).
                    id_order = {eid: idx for idx, eid in enumerate(event_ids)}
                    rows_sorted = sorted(rows, key=lambda r: id_order.get(r["id"], 999))

                    for row in rows_sorted:
                        payload = json.loads(row["payload"])
                        results.append({
                            "type": row["type"],
                            "source": row["source"],
                            "date": row["timestamp"],
                            # Include the snippet from the vector store (already
                            # extracted at indexing time) for consistency
                            "snippet": payload.get("snippet", payload.get("subject", ""))[:100],
                            # Include similarity score for debugging/transparency
                            "relevance": round(similarity_by_id.get(row["id"], 0.0), 3),
                        })
            except Exception as e:
                # Graceful degradation: if vector search fails for any reason
                # (database error, model loading issue, etc.), fall back to SQL.
                # This ensures search always returns something even if semantic
                # search is unavailable.
                logger.warning("Vector search failed, falling back to SQL LIKE: %s", e)
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
        # The system prompt is a synthesis guide for the search context assembled
        # above (query intent, current time, preferences, known semantic facts, mood
        # signals, and the ranked search results themselves).  It tells the LLM *how*
        # to use each context section — temporal anchoring, fact-based disambiguation,
        # result grouping, citation format, and no-result handling.  Without this
        # guide the LLM tends to just enumerate results chronologically or hallucinate
        # connections between unrelated events.
        system_prompt = """You are searching across the user's private digital life. Synthesize the results below into a direct, specific answer.

TEMPORAL REASONING — use the "Current time" section:
- Translate relative expressions in the query ("last month", "last week", "yesterday", "recently") to absolute dates using the current time before matching against result dates.
- Lead with the most recent relevant result when the query asks about something ongoing or evolving.

DISAMBIGUATION — use the "Known facts about user" section:
- Use known facts to resolve ambiguous references. If the query mentions "the project", "my work", "the meeting", or a first name, cross-reference facts to infer the most likely subject.
- Apply disambiguation silently — do not narrate the process or say "based on your facts…".

SYNTHESIS GUIDE:
1. Answer the question directly in the first sentence. Do not restate or rephrase the query.
2. Support the answer with 1-3 specific citations from the search results (date, source type, content).
3. If multiple results are relevant, group them thematically or chronologically rather than listing them arbitrarily.
4. If only one result matches, use it fully rather than padding with caveats.

CITATION FORMAT — for every factual claim include:
- The date (expressed relative to the current time, e.g. "3 days ago", "last Tuesday", "Jan 14")
- The source type (e.g. "in an email from Alice", "in a calendar event", "from a message", "in a task")
- A brief direct quote or close paraphrase of the relevant content

NO-RESULT HANDLING:
- If no search results appear below the context, say so in one sentence. Do not apologise or explain why.
- If results are only partially relevant, answer what the evidence supports and note the gap in one clause.

CONSTRAINTS:
- Output plain prose. Use bullet points only when listing 4 or more distinct items.
- Ground every statement in the search results. Never invent dates, names, or content.
- If mood context shows high stress, be maximally brief and direct — one short paragraph only."""

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
