"""Life OS v2 — AI engine.

Fresh-module rewrite of v1's ``services/ai_engine/engine.py``, trimmed to
the five method surface locked in the CEO plan and engineering review:

- :meth:`AIEngine.briefing_synthesis` — async, daily morning briefing.
- :meth:`AIEngine.task_extraction` — foreground, action-item JSON.
- :meth:`AIEngine.priority_classification` — foreground, single-word triage.
- :meth:`AIEngine.draft_reply` — background, per-contact ghostwriting.
- :meth:`AIEngine.semantic_search` — interactive, vector-backed recall.

What this module owns
---------------------
- **Routing.** Each call picks Ollama (local, default) or Anthropic
  Claude (cloud). Cloud is gated by ``use_cloud=True`` AND a non-empty
  ``cloud_api_key``; either false-y value silently falls back to local.
- **Per-operation budgets.** Every method wraps its inner query in
  :func:`asyncio.wait_for` with a method-specific timeout and raises
  :class:`AIBudgetExceeded` on overrun. The two numbers pinned by the
  CEO plan §1e (``task_extraction ≤ 2s`` foreground, ``briefing ≤ 20s``
  async) are literal constants here; the other three are picked to
  match the interaction profile the plan implies (priority is triage,
  draft waits briefly, search is user-facing).
- **Output parsing.** ``task_extraction`` strips markdown fences off
  the local model's reply before ``json.loads`` and returns ``[]`` on
  parse failure — graceful degradation preserved from v1 so the
  pipeline never stalls on a malformed LLM response. Same for
  ``priority_classification``: anything outside the four-word set
  coerces to ``"normal"``.

What this module deliberately does NOT own
------------------------------------------
- **Context assembly.** Callers pass in pre-assembled dicts /
  payloads. ``ai/context.py`` (next Week-7 task) builds them.
- **PII redaction.** Callers who want cloud + redaction wrap this
  engine with ``ai/pii.py`` (next Week-7 task). The engine itself
  sends the caller-supplied text to the cloud verbatim.
- **Mood / decision / expertise / values.** All four were killed in
  v2 (CEO plan § Killed Insights). No prompt here mentions them.
- **Vector-store embedding.** :meth:`semantic_search` hands off to
  an injected ``vector_store`` and synthesises a ranked list of
  :class:`SearchResult`. If no store is wired at construction, the
  method returns ``[]`` rather than raising — semantic search is
  non-critical for the main moment pipeline.

Constructor injection
---------------------
Follows the v2 convention locked in eng review §1a: every collaborator
is passed in; no global config dict. The LifeOS bootstrap is the
single site that wires these from ``config/settings.yaml``.

Budget hooks for tests
----------------------
Budgets are class constants so tests can override per-instance
(``engine.BUDGET_TASK_EXTRACTION = 0.05``) to drive
:class:`AIBudgetExceeded`. See ``tests/ai/test_engine.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# httpx is imported lazily inside _query_local / _query_cloud so unit
# tests that mock those two methods do not need the dependency
# installed; production callers do need it (pinned in requirements.txt).

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AIEngineError(Exception):
    """Transport / server-level error talking to the LLM backend.

    Carries a machine-readable ``error_type`` so operators can triage
    without parsing the message:

    - ``"connection"`` — Ollama / cloud endpoint unreachable.
    - ``"timeout"``    — raw HTTP timeout (distinct from budget overrun).
    - ``"server_error"`` — 4xx / 5xx response.
    - ``"bad_response"`` — response parsed, but unexpected shape.
    """

    def __init__(self, message: str, error_type: str, details: str = "") -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details


class AIBudgetExceeded(Exception):
    """Raised when an AI call exceeds its per-operation latency budget.

    The operation name (``"task_extraction"``, ``"briefing_synthesis"`` …)
    and the budget in seconds are both surfaced so callers can log or
    downgrade to a cached / stub response without parsing the message.
    """

    def __init__(self, operation: str, budget_seconds: float, elapsed_seconds: float) -> None:
        super().__init__(
            f"AI operation '{operation}' exceeded {budget_seconds:.2f}s budget (elapsed ~{elapsed_seconds:.2f}s)"
        )
        self.operation = operation
        self.budget_seconds = budget_seconds
        self.elapsed_seconds = elapsed_seconds


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One hit from :meth:`AIEngine.semantic_search`.

    Attributes
    ----------
    event_id:
        The ``events.id`` this hit points at. Stable across re-indexing
        since the chunk suffix (``_0``, ``_1``, …) is stripped upstream.
    event_type:
        The event's declared type (e.g. ``"email.received"``). Copied
        verbatim from the events table so producers can filter.
    source:
        The connector that emitted the event (``"proton_mail"`` …).
    timestamp:
        Unix seconds at event ingest.
    snippet:
        ≤200-char preview drawn from the event payload (subject /
        body). Not sanitised here; callers rendering to HTML must
        escape.
    score:
        Cosine similarity in ``[0, 1]``. Higher = closer match. 0.0
        means the score was unknown (non-vector fallback path).
    """

    event_id: str
    event_type: str
    source: str
    timestamp: int
    snippet: str
    score: float = 0.0


class VectorStore(Protocol):
    """Duck-typed slice of the v2 vector store the engine actually uses.

    Only one method is called here, mirroring v1's ``VectorStore.search``.
    Keeping the protocol local avoids a hard import cycle with
    :mod:`storage.vector_store` and lets tests pass a minimal fake.
    """

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)
_ALLOWED_PRIORITIES = frozenset({"critical", "high", "normal", "low"})


class AIEngine:
    """Async LLM orchestrator: Ollama local + optional Claude cloud.

    See the module docstring for the motivation. This class is
    stateless between calls — each method assembles its prompt,
    dispatches, enforces its budget, and returns. Constructor
    parameters are stored verbatim; no heavy work happens at init
    time.
    """

    # --- per-operation latency budgets (seconds) ----------------------------
    # Two are pinned by the CEO plan §1e. The other three are defaults
    # picked to match the interaction profile in DESIGN.md: priority
    # triage is foreground (2s), draft reply is "user waits briefly"
    # (10s), semantic search is interactive (5s). Tests patch these on
    # the instance to drive AIBudgetExceeded.
    BUDGET_BRIEFING: float = 20.0
    BUDGET_TASK_EXTRACTION: float = 2.0
    BUDGET_PRIORITY: float = 2.0
    BUDGET_DRAFT_REPLY: float = 10.0
    BUDGET_SEMANTIC_SEARCH: float = 5.0

    # Outer cap on the raw HTTP call. The budget timeout trips first;
    # this is only a safety net so a wedged TCP connection does not
    # leak a pending task.
    HTTP_TIMEOUT_SECONDS: float = 120.0

    # Cloud defaults. Model id is the Claude 4.x Sonnet checkpoint
    # locked in the CEO plan; callers can override.
    DEFAULT_CLOUD_MODEL: str = "claude-sonnet-4-5-20250514"
    CLOUD_MAX_TOKENS: int = 2048

    def __init__(
        self,
        ollama_url: str,
        ollama_model: str,
        *,
        cloud_api_key: str | None = None,
        use_cloud: bool = False,
        cloud_model: str | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Wire the engine.

        Parameters
        ----------
        ollama_url:
            Base URL of the local Ollama server (e.g.
            ``"http://localhost:11434"``). No trailing slash required.
        ollama_model:
            Ollama model tag (``"mistral"``, ``"llama3.2"``, …).
        cloud_api_key:
            Anthropic API key. ``None`` disables the cloud path.
        use_cloud:
            Opt-in toggle. The cloud path is active iff this is
            ``True`` AND ``cloud_api_key`` is a non-empty string. This
            double-gate matches v1's "cloud is always an explicit
            choice" convention.
        cloud_model:
            Optional override for the Claude model id. Defaults to
            :data:`DEFAULT_CLOUD_MODEL`.
        vector_store:
            Optional vector store used by :meth:`semantic_search`.
            Absence is not an error — the method returns ``[]``.
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.cloud_api_key = cloud_api_key
        self.cloud_model = cloud_model or self.DEFAULT_CLOUD_MODEL
        self.use_cloud = bool(use_cloud and cloud_api_key)
        self.vector_store = vector_store

    # ---------------- public operations -----------------------------------

    async def briefing_synthesis(self, context: dict[str, Any]) -> str:
        """Synthesise the morning briefing from a pre-assembled context.

        Parameters
        ----------
        context:
            Dict produced by ``ai/context.assemble_briefing_context``
            (not in this module). The full shape is JSON-serialised
            into the user message; no keys are required at the engine
            level.

        Returns
        -------
        str
            Plain-prose briefing.

        Raises
        ------
        AIBudgetExceeded
            If the call exceeds :data:`BUDGET_BRIEFING`.
        AIEngineError
            On transport failure.
        """
        system_prompt = (
            "You are a private personal assistant generating a personalised morning "
            "briefing. Ground every sentence in the provided context. Never invent "
            "names, dates, tasks, or events. Output plain prose only — no section "
            "headers or labels. Do not reference mood, decisions, expertise, or "
            "values; those signals are not available in v2."
        )
        user_prompt = json.dumps(context, default=str, sort_keys=True)
        return await self._query_with_budget(
            operation="briefing_synthesis",
            budget=self.BUDGET_BRIEFING,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cloud_preferred=False,
        )

    async def task_extraction(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract action items from a single event payload.

        Parameters
        ----------
        event:
            Event envelope. ``payload.subject`` + ``payload.snippet`` +
            ``payload.body`` are concatenated into the user message;
            missing keys default to empty string.

        Returns
        -------
        list of dict
            Action items with keys ``title`` (str), ``due_hint``
            (str | None), ``priority`` (``"high"|"normal"|"low"``),
            ``completed`` (bool). Returns ``[]`` on any parse failure
            — the pipeline must not stall on a malformed LLM response.

        Raises
        ------
        AIBudgetExceeded
            If the call exceeds :data:`BUDGET_TASK_EXTRACTION`.
        """
        system_prompt = (
            "You are a task extraction specialist. Identify genuine action items "
            "that the message recipient must personally complete. Skip marketing, "
            "automated notifications, and tasks assigned to others. Mark "
            "'completed': true for past-tense reports of finished work. "
            "Return ONLY a valid JSON array of objects with keys "
            '"title" (string), "due_hint" (string or null), '
            '"priority" ("high"|"normal"|"low"), "completed" (true|false). '
            "Return [] if there are no action items. No markdown fences, no prose."
        )
        user_prompt = self._event_to_text(event)
        raw = await self._query_with_budget(
            operation="task_extraction",
            budget=self.BUDGET_TASK_EXTRACTION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cloud_preferred=False,
        )
        return _parse_task_list(raw)

    async def priority_classification(self, event: dict[str, Any]) -> str:
        """Classify the priority of an incoming event.

        Parameters
        ----------
        event:
            Event envelope. Uses ``payload.from_address``,
            ``payload.subject``, and the first 200 chars of
            ``payload.snippet`` / ``payload.body``.

        Returns
        -------
        str
            One of ``"critical"``, ``"high"``, ``"normal"``, ``"low"``.
            Defaults to ``"normal"`` when the LLM returns anything
            unrecognised.

        Raises
        ------
        AIBudgetExceeded
            If the call exceeds :data:`BUDGET_PRIORITY`.
        """
        payload = event.get("payload") or {}
        from_addr = str(payload.get("from_address") or "")
        subject = str(payload.get("subject") or "")
        snippet = str(payload.get("snippet") or payload.get("body") or "")[:200]
        user_prompt = (
            f"From: {from_addr}\nSubject: {subject}\nPreview: {snippet}\n\n"
            "Respond with exactly one word: critical, high, normal, or low."
        )
        system_prompt = "You are a message priority classifier. Respond with exactly one word."
        raw = await self._query_with_budget(
            operation="priority_classification",
            budget=self.BUDGET_PRIORITY,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cloud_preferred=False,
        )
        word = raw.strip().lower().split()[0] if raw.strip() else ""
        return word if word in _ALLOWED_PRIORITIES else "normal"

    async def draft_reply(
        self,
        contact_id: str,
        recent_messages: list[dict[str, Any]],
        user_style: dict[str, Any],
    ) -> str:
        """Ghostwrite a reply in the user's voice.

        Routes to the cloud when :attr:`use_cloud` is set (Claude draft
        quality beats local Mistral by a wide margin; PII stripping is
        the caller's job — see ``ai/pii.py``). Falls back to Ollama
        transparently when the cloud path is disabled.

        Parameters
        ----------
        contact_id:
            Opaque contact identifier (resolved upstream).
        recent_messages:
            Thread history (newest-last). Each dict at minimum:
            ``direction`` (``"inbound"|"outbound"``), ``body`` (str).
        user_style:
            Per-contact style profile from
            ``signal_profiles[producer='comm_template']``. Keys read
            here: ``contact_name``, ``channel``, ``template_style``,
            ``formality``, ``typical_length``. Unknown keys are
            ignored.

        Returns
        -------
        str
            Draft message body only. No preamble, no labels.

        Raises
        ------
        AIBudgetExceeded
            If the call exceeds :data:`BUDGET_DRAFT_REPLY`.
        """
        system_prompt = (
            "You are ghostwriting a reply on behalf of the user. Output ONLY the "
            "message text — no preamble, no labels, no meta-commentary. Match the "
            "user's communication style from the provided profile (formality, "
            "length, greeting, closing). If the incoming message is a question, "
            "answer it directly. Ground every sentence in the provided context; "
            "never invent facts or dates."
        )
        user_prompt = json.dumps(
            {
                "contact_id": contact_id,
                "user_style": user_style,
                "recent_messages": recent_messages,
            },
            default=str,
            sort_keys=True,
        )
        return await self._query_with_budget(
            operation="draft_reply",
            budget=self.BUDGET_DRAFT_REPLY,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cloud_preferred=True,
        )

    async def semantic_search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top-``k`` semantically similar events.

        Delegates the actual vector lookup to the injected
        ``vector_store``. When no store is wired, returns ``[]`` —
        semantic search is non-critical for the moment pipeline.

        The vector store is expected to return a list of dicts with
        keys ``doc_id`` and ``score``. Chunk suffixes (``_0``, ``_1``,
        …) on ``doc_id`` are stripped to recover the event id, and
        the best score per event is kept when multiple chunks match.
        Payload / metadata may also appear on each hit; it is copied
        through when present.

        Parameters
        ----------
        query:
            Natural-language query string.
        k:
            Max results. Values ≤ 0 coerce to 1.

        Returns
        -------
        list of SearchResult
            Sorted by ``score`` descending.

        Raises
        ------
        AIBudgetExceeded
            If the call exceeds :data:`BUDGET_SEMANTIC_SEARCH`.
        """
        if self.vector_store is None:
            return []
        k = max(1, k)

        async def _run() -> list[SearchResult]:
            # The vector store is synchronous; off-load to a worker
            # thread so the event loop stays responsive and the outer
            # wait_for can still cancel on budget overrun.
            raw_hits = await asyncio.to_thread(self.vector_store.search, query, k * 2)
            return _normalise_search_hits(raw_hits, k)

        started = time.monotonic()
        try:
            return await asyncio.wait_for(_run(), timeout=self.BUDGET_SEMANTIC_SEARCH)
        except TimeoutError as exc:
            elapsed = time.monotonic() - started
            raise AIBudgetExceeded("semantic_search", self.BUDGET_SEMANTIC_SEARCH, elapsed) from exc

    # ---------------- internal helpers ------------------------------------

    async def _query_with_budget(
        self,
        *,
        operation: str,
        budget: float,
        system_prompt: str,
        user_prompt: str,
        cloud_preferred: bool,
    ) -> str:
        """Dispatch one LLM call under a per-operation timeout.

        ``cloud_preferred`` controls the routing:

        - ``True`` + ``use_cloud`` → Claude, fall through to Ollama
          only if the cloud path is disabled.
        - ``False`` → always Ollama; the cloud is not consulted even
          when ``use_cloud`` is True. Keeps briefing + task extraction
          + priority triage on-device (v1 convention).
        """
        started = time.monotonic()
        try:
            if cloud_preferred and self.use_cloud:
                coro = self._query_cloud(system_prompt, user_prompt)
            else:
                coro = self._query_local(system_prompt, user_prompt)
            return await asyncio.wait_for(coro, timeout=budget)
        except TimeoutError as exc:
            elapsed = time.monotonic() - started
            raise AIBudgetExceeded(operation, budget, elapsed) from exc

    async def _query_local(self, system_prompt: str, user_prompt: str) -> str:
        """POST to Ollama's ``/api/chat`` endpoint.

        Returns the assistant's first message content. Raises
        :class:`AIEngineError` on transport-level failure (see the
        class docstring for the four error_type categories).
        """
        import httpx

        url = f"{self.ollama_url}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
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
                content = data.get("message", {}).get("content")
                if not isinstance(content, str):
                    raise AIEngineError(
                        "Ollama returned unexpected payload shape",
                        "bad_response",
                        f"message.content not a string: {type(content).__name__}",
                    )
                return content
        except httpx.ConnectError as exc:
            raise AIEngineError(
                "Ollama service unreachable",
                "connection",
                f"Could not connect to {self.ollama_url}",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineError(
                "Ollama HTTP call timed out",
                "timeout",
                f"Exceeded {self.HTTP_TIMEOUT_SECONDS}s HTTP timeout",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise AIEngineError(
                f"Ollama returned HTTP {exc.response.status_code}",
                "server_error",
                str(exc),
            ) from exc

    async def _query_cloud(self, system_prompt: str, user_prompt: str) -> str:
        """POST to Anthropic's Messages API.

        Transparent fall-through to :meth:`_query_local` when
        ``cloud_api_key`` is unset so callers never need to branch on
        config. Caller-supplied ``user_prompt`` is sent verbatim; PII
        redaction is NOT the engine's job (see module docstring).
        """
        if not self.cloud_api_key:
            return await self._query_local(system_prompt, user_prompt)
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.cloud_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.cloud_model,
                        "max_tokens": self.CLOUD_MAX_TOKENS,
                        "system": system_prompt,
                        "messages": [
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                )
                response.raise_for_status()
                data = response.json()
                blocks = data.get("content") or []
                if blocks and isinstance(blocks, list):
                    first = blocks[0]
                    if isinstance(first, dict) and first.get("type") == "text":
                        text = first.get("text")
                        if isinstance(text, str):
                            return text
                raise AIEngineError(
                    "Claude returned no text content block",
                    "bad_response",
                    f"content shape: {type(blocks).__name__}",
                )
        except httpx.ConnectError as exc:
            raise AIEngineError(
                "Anthropic API unreachable",
                "connection",
                "Could not connect to api.anthropic.com",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIEngineError(
                "Anthropic API call timed out",
                "timeout",
                f"Exceeded {self.HTTP_TIMEOUT_SECONDS}s HTTP timeout",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise AIEngineError(
                f"Anthropic returned HTTP {exc.response.status_code}",
                "server_error",
                str(exc),
            ) from exc

    @staticmethod
    def _event_to_text(event: dict[str, Any]) -> str:
        """Flatten an event envelope into the text the LLM sees.

        Kept separate so tests can exercise the shaping without a
        live LLM round-trip.
        """
        payload = event.get("payload") or {}
        parts = [
            f"Subject: {payload.get('subject', '')}",
            f"From: {payload.get('from_address', '')}",
            f"Snippet: {payload.get('snippet', '')}",
        ]
        body = payload.get("body")
        if body:
            parts.append(f"Body: {body}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Free helpers (exposed for tests)
# ---------------------------------------------------------------------------


def _parse_task_list(raw: str) -> list[dict[str, Any]]:
    """Parse the model's task-extraction output into a list of dicts.

    Strips surrounding ``” ```json / ``` ”`` fences, then runs
    ``json.loads``. Any structural failure (not a list, non-dict
    elements) drops the output and returns ``[]`` — pipeline must
    not stall on a malformed reply.
    """
    cleaned = _CODE_FENCE.sub("", raw).strip()
    if not cleaned:
        return []
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("task_extraction JSON parse failed; dropping output")
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    return out


def _normalise_search_hits(raw_hits: list[dict[str, Any]], k: int) -> list[SearchResult]:
    """Collapse chunk suffixes, dedupe by event id, return top-k.

    The vector store returns one row per chunk; a 3-chunk email
    yields three hits. We strip the ``_<digits>`` suffix to recover
    the event id and keep the best score for each event.
    """
    best: dict[str, _HitAccumulator] = {}
    for hit in raw_hits:
        if not isinstance(hit, dict):
            continue
        doc_id = hit.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        parts = doc_id.rsplit("_", 1)
        event_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else doc_id
        score = _coerce_float(hit.get("score"))
        snippet = str(hit.get("snippet") or hit.get("subject") or "")[:200]
        event_type = str(hit.get("event_type") or hit.get("type") or "")
        source = str(hit.get("source") or "")
        timestamp = _coerce_int(hit.get("timestamp"))
        existing = best.get(event_id)
        if existing is None or score > existing.score:
            best[event_id] = _HitAccumulator(
                event_id=event_id,
                event_type=event_type or (existing.event_type if existing else ""),
                source=source or (existing.source if existing else ""),
                timestamp=timestamp or (existing.timestamp if existing else 0),
                snippet=snippet or (existing.snippet if existing else ""),
                score=score,
            )
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
    return [
        SearchResult(
            event_id=h.event_id,
            event_type=h.event_type,
            source=h.source,
            timestamp=h.timestamp,
            snippet=h.snippet,
            score=h.score,
        )
        for h in ranked
    ]


@dataclass(slots=True)
class _HitAccumulator:
    event_id: str
    event_type: str
    source: str
    timestamp: int
    snippet: str
    score: float = field(default=0.0)


def _coerce_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
