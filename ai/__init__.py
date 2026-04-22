"""Life OS v2 — AI package.

Fresh-module rewrite of v1's ``services/ai_engine``. Three public
symbols live in :mod:`ai.engine` (AIEngine + AIBudgetExceeded +
SearchResult); the PII shield and context assembly will land in
sibling modules in later Week-7 tasks (see ``NEXT_TASKS.md``).

The package owns LLM wire-level concerns only: Ollama + Claude
routing, per-operation latency budgets, and JSON / priority parsing.
Context assembly (what to feed each call) lives in :mod:`ai.context`;
PII redaction (what to strip before cloud calls) lives in
:mod:`ai.pii`. Those modules are still under construction; this
one stands alone.
"""

from ai.engine import (
    AIBudgetExceeded,
    AIEngine,
    AIEngineError,
    SearchResult,
)

__all__ = [
    "AIBudgetExceeded",
    "AIEngine",
    "AIEngineError",
    "SearchResult",
]
