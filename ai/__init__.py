"""Life OS v2 — AI package.

Fresh-module rewrite of v1's ``services/ai_engine``. Public symbols
live in :mod:`ai.engine` (AIEngine + AIBudgetExceeded +
SearchResult) and :mod:`ai.pii` (PIIShield). Context assembly will
land in :mod:`ai.context` in a later Week-7 task (see
``NEXT_TASKS.md``).

The package owns LLM wire-level concerns only: Ollama + Claude
routing, per-operation latency budgets, JSON / priority parsing, and
PII redaction for cloud calls. Context assembly (what to feed each
call) lives in :mod:`ai.context` and is still under construction.
"""

from ai.engine import (
    AIBudgetExceeded,
    AIEngine,
    AIEngineError,
    SearchResult,
)
from ai.pii import PIIShield

__all__ = [
    "AIBudgetExceeded",
    "AIEngine",
    "AIEngineError",
    "PIIShield",
    "SearchResult",
]
