"""`core.moment` — the Moment primitive.

The Moment is the single user-facing unit in Life OS v2: a bundle of
``(time + context + insight + evidence + proposed_action + state)`` that
carries provenance and a decision loop. Predictions, rules, tasks, and
notifications from v1 all collapse into this one type.

Reference: CEO plan at
``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
section "The Moment Primitive", and engineering plan at
``docs/plans/2026-04-21-v2-rewrite-plan.md``.

This package is split into:

- ``types`` — enums + dataclasses (this scaffold task)
- ``state`` — state machine + legal-transition table
- ``producer`` — abstract producer base + registry
- ``scheduler`` — wall-clock firing loop
- ``engine`` — orchestration over producers, repos, feedback weights
"""

from core.moment.types import (
    Action,
    ActionKind,
    ContextTrigger,
    InsightType,
    Moment,
    MomentState,
    StateHistoryEntry,
)

__all__ = [
    "Action",
    "ActionKind",
    "ContextTrigger",
    "InsightType",
    "Moment",
    "MomentState",
    "StateHistoryEntry",
]
