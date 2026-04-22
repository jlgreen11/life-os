"""Moment primitive — enums and dataclasses.

This module holds the **pure data types** for the Moment primitive. No
business logic lives here: no state transitions, no persistence, no action
dispatch. The state machine is defined alongside in ``core/moment/state.py``;
persistence in ``storage/repos/moments.py``.

Enum string values mirror the SQLite ``CHECK`` constraints declared in
``storage/schema.py`` (the ``moments`` and ``moment_state_history`` tables).
Mismatches here are schema bugs — tests assert the full set of members.

Using ``StrEnum`` (Python 3.11+) makes the members JSON-serializable out
of the box, so ``json.dumps(dataclasses.asdict(moment))`` works without a
custom encoder.

References
----------
CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
section "The Moment Primitive (new data model)".

Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MomentState(StrEnum):
    """Lifecycle state for a Moment.

    Mirrors the ``state`` column ``CHECK`` constraint on the ``moments``
    table in ``storage/schema.py``. Legal transitions are defined in
    ``core/moment/state.py``; any attempt to set an illegal state goes
    through ``validate_transition`` and raises ``IllegalTransition``.

    ``dismissed``, ``done``, and ``expired`` are terminal (CEO plan §
    "State-machine transitions").
    """

    SUGGESTED = "suggested"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    SNOOZED = "snoozed"
    DONE = "done"
    EXPIRED = "expired"


class InsightType(StrEnum):
    """Producer identity — which signal extractor emitted a Moment.

    The full Phase 1 set is fixed at six (CEO plan § "The Moment Primitive").
    The schema also allows ``legacy_task`` as a migration escape hatch for
    v1 tasks pulled forward during cutover; that value lives in the DB
    ``CHECK`` constraint but is intentionally **not** a producer insight
    type and therefore not enumerated here. Consumers that need it can
    special-case the string directly.
    """

    CADENCE = "cadence"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    COMM_TEMPLATE = "comm_template"
    ROUTINE = "routine"


class ActionKind(StrEnum):
    """What a Moment proposes to do when accepted.

    Action semantics live in the outbox dispatcher (Week 11). Here we only
    name the kinds the UI can render and the producer layer can emit. See
    CEO plan § "The Moment Primitive → Enums".
    """

    DRAFT_MESSAGE = "draft_message"
    SEND_MESSAGE = "send_message"
    SCHEDULE_BLOCK = "schedule_block"
    ARCHIVE_EVENT = "archive_event"
    NUDGE = "nudge"
    SET_REMINDER = "set_reminder"
    CREATE_CALENDAR_ENTRY = "create_calendar_entry"
    NOTE_OBSERVATION = "note_observation"


@dataclass
class Action:
    """A proposed side effect bundled onto a Moment.

    ``params`` is a free-form mapping whose schema is per-``kind`` and
    validated by the outbox dispatcher, not here. Keeping the mapping
    open avoids a combinatorial explosion of per-kind dataclasses while
    still making the data JSON-serializable via ``dataclasses.asdict``.

    CEO plan: ``Action { kind: ActionKind, params: { ... } }``.
    """

    kind: ActionKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextTrigger:
    """A compact expression declaring when a Moment should fire.

    Grammar (CEO plan § "ContextTrigger grammar (v1 vocabulary)"):

    - ``calendar:gap>{minutes}m``
    - ``calendar:before_event:{minutes}m``
    - ``calendar:after_event:{minutes}m``
    - ``arrive:{place}``
    - ``depart:{place}``
    - ``time:{HH:MM}``
    - ``weekday:{day}``
    - ``after_inactivity:{hours}h:{channel}``
    - ``event_type:{type}``

    Parsing and evaluation live in ``core/moment/scheduler.py`` (Week 3).
    This dataclass is just the transport — validation is deferred so that
    scaffolding does not block on grammar decisions still in play.
    """

    expression: str


@dataclass
class StateHistoryEntry:
    """One row of a Moment's transition audit log.

    Mirrors ``moment_state_history`` (``storage/schema.py``). Kept as a
    dataclass so in-memory Moments can carry history without a round-trip
    to SQLite, e.g. during producer-level replay or tests.
    """

    from_state: MomentState | None
    to_state: MomentState
    ts: int
    annotation: str | None = None


@dataclass
class Moment:
    """A single evidence-backed, state-machine-governed user-facing unit.

    Fields mirror ``storage/schema.py::CREATE_MOMENTS_SQL`` one-for-one, so
    a repository can round-trip a Moment through SQLite without lossy
    serialization.

    Defaults
    --------
    - ``state`` defaults to ``SUGGESTED`` — the only legal entry state.
    - ``evidence`` and ``state_history`` default to empty lists (not
      ``None``) so callers never need to null-check.
    - ``confidence`` and ``feedback_weight`` default to the same values
      as the SQLite columns (``0.0`` and ``1.0`` respectively).
    - ``expires_at`` has no default here: producers must set it explicitly
      (CEO plan: default ``created_at + 72h`` unless overridden).

    References
    ----------
    CEO plan § "The Moment Primitive (new data model)".
    """

    id: str
    created_at: int
    expires_at: int
    insight: str
    evidence_hash: str
    proposed_action: Action
    source_insight_type: InsightType
    scheduled_for: int | None = None
    context_trigger: ContextTrigger | None = None
    evidence: list[str] = field(default_factory=list)
    state: MomentState = MomentState.SUGGESTED
    state_history: list[StateHistoryEntry] = field(default_factory=list)
    snooze_until: int | None = None
    confidence: float = 0.0
    feedback_weight: float = 1.0
