"""Life OS v2 — AI package.

Fresh-module rewrite of v1's ``services/ai_engine``. Public symbols
live in :mod:`ai.engine` (AIEngine + AIBudgetExceeded +
SearchResult), :mod:`ai.pii` (PIIShield), and :mod:`ai.context`
(ContextAssembler + BRIEFING_SECTIONS).

The package owns LLM wire-level concerns only: Ollama + Claude
routing, per-operation latency budgets, JSON / priority parsing,
PII redaction for cloud calls, and briefing context assembly.
"""

from ai.context import BRIEFING_SECTIONS, ContextAssembler
from ai.engine import (
    AIBudgetExceeded,
    AIEngine,
    AIEngineError,
    SearchResult,
)
from ai.pii import PIIShield

__all__ = [
    "BRIEFING_SECTIONS",
    "AIBudgetExceeded",
    "AIEngine",
    "AIEngineError",
    "ContextAssembler",
    "PIIShield",
    "SearchResult",
]
