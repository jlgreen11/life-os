"""Tests for ``core.moment.types``.

Asserts:

1. Every enum has the full set of members named in the CEO plan and they
   carry the exact string values the SQLite ``CHECK`` constraints expect.
2. ``Moment`` instantiates with sensible defaults (state ``SUGGESTED``,
   empty lists, 0.0/1.0 confidence/feedback_weight).
3. A fully populated Moment round-trips through
   ``dataclasses.asdict → json.dumps → json.loads`` — i.e. the dataclass
   graph is natively JSON-serializable because the enums mix in ``str``.
"""

from __future__ import annotations

import dataclasses
import json

from core.moment.types import (
    Action,
    ActionKind,
    ContextTrigger,
    InsightType,
    Moment,
    MomentState,
    StateHistoryEntry,
)


def test_moment_state_members_match_schema_check_constraint():
    """Enum values must be exactly the strings the DB CHECK constraint uses."""
    assert {m.value for m in MomentState} == {
        "suggested",
        "accepted",
        "dismissed",
        "snoozed",
        "done",
        "expired",
    }


def test_moment_state_member_names():
    """Names (Python identifiers) match the CEO-plan capitalization."""
    assert {m.name for m in MomentState} == {
        "SUGGESTED",
        "ACCEPTED",
        "DISMISSED",
        "SNOOZED",
        "DONE",
        "EXPIRED",
    }


def test_insight_type_members():
    """Six insight types, matching the CEO plan's Phase 1 set."""
    assert {i.value for i in InsightType} == {
        "cadence",
        "relationship",
        "temporal",
        "spatial",
        "comm_template",
        "routine",
    }


def test_action_kind_members():
    """All eight ActionKinds the CEO plan enumerates."""
    assert {a.value for a in ActionKind} == {
        "draft_message",
        "send_message",
        "schedule_block",
        "archive_event",
        "nudge",
        "set_reminder",
        "create_calendar_entry",
        "note_observation",
    }


def test_action_params_default_is_empty_dict():
    """Action().params defaults to {} — callers never need to null-check."""
    a = Action(kind=ActionKind.NUDGE)
    assert a.params == {}


def test_action_params_defaults_are_not_shared():
    """Mutable default must not be shared across instances."""
    a1 = Action(kind=ActionKind.NUDGE)
    a2 = Action(kind=ActionKind.NUDGE)
    a1.params["x"] = 1
    assert a2.params == {}


def test_context_trigger_is_a_plain_wrapper():
    """ContextTrigger carries the expression string verbatim — no parsing yet."""
    t = ContextTrigger(expression="calendar:gap>30m")
    assert t.expression == "calendar:gap>30m"


def test_moment_defaults():
    """A minimally constructed Moment has the CEO-plan-specified defaults."""
    m = Moment(
        id="m1",
        created_at=1_700_000_000,
        expires_at=1_700_259_200,
        insight="placeholder insight",
        evidence_hash="abc123",
        proposed_action=Action(kind=ActionKind.NUDGE),
        source_insight_type=InsightType.CADENCE,
    )
    assert m.state is MomentState.SUGGESTED
    assert m.evidence == []
    assert m.state_history == []
    assert m.scheduled_for is None
    assert m.context_trigger is None
    assert m.snooze_until is None
    assert m.confidence == 0.0
    assert m.feedback_weight == 1.0


def test_moment_default_lists_are_not_shared():
    """Mutable-default lists must be per-instance."""
    m1 = Moment(
        id="m1",
        created_at=0,
        expires_at=1,
        insight="",
        evidence_hash="h1",
        proposed_action=Action(kind=ActionKind.NUDGE),
        source_insight_type=InsightType.CADENCE,
    )
    m2 = Moment(
        id="m2",
        created_at=0,
        expires_at=1,
        insight="",
        evidence_hash="h2",
        proposed_action=Action(kind=ActionKind.NUDGE),
        source_insight_type=InsightType.CADENCE,
    )
    m1.evidence.append("event-1")
    m1.state_history.append(StateHistoryEntry(from_state=None, to_state=MomentState.SUGGESTED, ts=0))
    assert m2.evidence == []
    assert m2.state_history == []


def test_moment_json_round_trip_via_asdict():
    """asdict → json.dumps → json.loads preserves every field verbatim.

    This is the contract the NEXT_TASKS scaffold task asks for: StrEnum
    makes the enum members JSON-native, so dataclasses.asdict produces a
    tree json.dumps can serialize with no custom encoder.
    """
    original = Moment(
        id="m-round-trip",
        created_at=1_700_000_000,
        scheduled_for=1_700_003_600,
        expires_at=1_700_259_200,
        context_trigger=ContextTrigger(expression="calendar:gap>30m"),
        insight="Historical focus window at 9am; 45-min gap open now.",
        evidence=["evt-a", "evt-b", "evt-c"],
        evidence_hash="deadbeef",
        proposed_action=Action(
            kind=ActionKind.SCHEDULE_BLOCK,
            params={"duration_minutes": 45, "label": "deep work"},
        ),
        state=MomentState.SUGGESTED,
        state_history=[
            StateHistoryEntry(
                from_state=None,
                to_state=MomentState.SUGGESTED,
                ts=1_700_000_000,
                annotation="created",
            ),
        ],
        snooze_until=None,
        confidence=0.72,
        feedback_weight=0.88,
        source_insight_type=InsightType.TEMPORAL,
    )

    as_dict = dataclasses.asdict(original)
    serialized = json.dumps(as_dict)  # must not raise
    round_tripped = json.loads(serialized)

    # Enums became their string values (str-Enum mix-in is how).
    assert round_tripped["state"] == "suggested"
    assert round_tripped["source_insight_type"] == "temporal"
    assert round_tripped["proposed_action"]["kind"] == "schedule_block"
    assert round_tripped["proposed_action"]["params"] == {
        "duration_minutes": 45,
        "label": "deep work",
    }
    # Nested structures survive intact.
    assert round_tripped["evidence"] == ["evt-a", "evt-b", "evt-c"]
    assert round_tripped["context_trigger"] == {"expression": "calendar:gap>30m"}
    assert round_tripped["state_history"] == [
        {
            "from_state": None,
            "to_state": "suggested",
            "ts": 1_700_000_000,
            "annotation": "created",
        }
    ]
    # Primitives survive.
    assert round_tripped["confidence"] == 0.72
    assert round_tripped["feedback_weight"] == 0.88
    assert round_tripped["scheduled_for"] == 1_700_003_600
    assert round_tripped["snooze_until"] is None


def test_state_history_entry_defaults():
    """StateHistoryEntry.annotation defaults to None (optional field)."""
    e = StateHistoryEntry(
        from_state=MomentState.SUGGESTED,
        to_state=MomentState.ACCEPTED,
        ts=1_700_000_000,
    )
    assert e.annotation is None


def test_public_reexports_match_types_module():
    """`core.moment` package re-exports every type the scaffold publishes."""
    from core import moment as pkg

    expected = {
        "Action",
        "ActionKind",
        "ContextTrigger",
        "InsightType",
        "Moment",
        "MomentState",
        "StateHistoryEntry",
    }
    for name in expected:
        assert hasattr(pkg, name), f"core.moment missing re-export: {name}"
