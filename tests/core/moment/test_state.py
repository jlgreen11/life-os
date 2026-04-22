"""Tests for ``core.moment.state``.

The state machine is authoritative for all Moment lifecycle transitions,
so these tests enumerate the **entire** cartesian product of
``(from_state, to_state)`` — including ``None`` on the ``from`` side to
cover creation — and split the product into two disjoint sets:

1. pairs listed in ``_LEGAL_TRANSITIONS`` must pass through
   :func:`validate_transition` silently;
2. every other pair must raise :class:`IllegalTransition`.

Driving both halves off the same single source of truth (the table
itself) means adding or removing an edge automatically widens/narrows
the expected-to-raise set, so drift between the table and the tests is
impossible without intentionally editing the table.

The ``to`` side never includes ``None`` because the API does not model
a transition into "not created" — deletion is separate, and the
contract of ``validate_transition`` requires a concrete target.
"""

from __future__ import annotations

import itertools

import pytest

from core.moment.state import (
    _LEGAL_TRANSITIONS,
    IllegalTransition,
    validate_transition,
)
from core.moment.types import MomentState

_ALL_FROM_STATES: tuple[MomentState | None, ...] = (None, *MomentState)
_ALL_TO_STATES: tuple[MomentState, ...] = tuple(MomentState)


def _legal_pairs() -> list[tuple[MomentState | None, MomentState]]:
    """Flatten ``_LEGAL_TRANSITIONS`` into ``(from, to)`` pairs."""
    pairs: list[tuple[MomentState | None, MomentState]] = []
    for src, targets in _LEGAL_TRANSITIONS.items():
        for tgt in targets:
            pairs.append((src, tgt))
    return pairs


def _illegal_pairs() -> list[tuple[MomentState | None, MomentState]]:
    """Every ``(from, to)`` in the product that is NOT in the legal set."""
    legal = set(_legal_pairs())
    return [pair for pair in itertools.product(_ALL_FROM_STATES, _ALL_TO_STATES) if pair not in legal]


def test_legal_transition_table_matches_ceo_plan() -> None:
    """Guard against accidental edits to the legal-transition table.

    If this test needs updating, the CEO plan's "State-machine
    transitions" section should have changed first.
    """
    assert _LEGAL_TRANSITIONS == {
        None: {MomentState.SUGGESTED},
        MomentState.SUGGESTED: {
            MomentState.ACCEPTED,
            MomentState.DISMISSED,
            MomentState.SNOOZED,
            MomentState.EXPIRED,
        },
        MomentState.ACCEPTED: {MomentState.DONE},
        MomentState.SNOOZED: {MomentState.SUGGESTED, MomentState.EXPIRED},
        MomentState.DISMISSED: set(),
        MomentState.DONE: set(),
        MomentState.EXPIRED: set(),
    }


def test_every_state_appears_as_a_key() -> None:
    """Every MomentState (plus None) must be a table key — no silent omissions.

    Without this, an added enum member would pass through
    :func:`validate_transition` as an ``.get(..., set())`` miss and
    look merely "terminal" when it is actually "unspecified".
    """
    assert set(_LEGAL_TRANSITIONS.keys()) == {None, *MomentState}


@pytest.mark.parametrize(("from_state", "to_state"), _legal_pairs())
def test_legal_transitions_pass(from_state: MomentState | None, to_state: MomentState) -> None:
    """Every legal pair from the table validates silently."""
    # Must not raise.
    assert validate_transition(from_state, to_state) is None


@pytest.mark.parametrize(("from_state", "to_state"), _illegal_pairs())
def test_illegal_transitions_raise(from_state: MomentState | None, to_state: MomentState) -> None:
    """Every non-legal pair in the cartesian product raises IllegalTransition."""
    with pytest.raises(IllegalTransition):
        validate_transition(from_state, to_state)


def test_illegal_transition_is_a_value_error() -> None:
    """IllegalTransition must subclass ValueError for existing handlers.

    Repo and route layers already catch ``ValueError`` around input
    validation; the state-machine error piggy-backs on that path.
    """
    assert issubclass(IllegalTransition, ValueError)


def test_illegal_transition_message_includes_states_and_legal_set() -> None:
    """Error message must be actionable — name the attempted pair and targets."""
    with pytest.raises(IllegalTransition) as excinfo:
        validate_transition(MomentState.DONE, MomentState.SUGGESTED)
    msg = str(excinfo.value)
    # Names the attempted pair so a reader can find the offending call.
    assert "done" in msg.lower()
    assert "suggested" in msg.lower()
    # DONE is terminal — message must communicate that there's nowhere to go.
    assert "terminal" in msg.lower()


def test_illegal_transition_message_lists_allowed_targets() -> None:
    """For a non-terminal from_state, the error lists the real legal targets."""
    with pytest.raises(IllegalTransition) as excinfo:
        validate_transition(MomentState.SUGGESTED, MomentState.DONE)
    msg = str(excinfo.value)
    # SUGGESTED → {ACCEPTED, DISMISSED, SNOOZED, EXPIRED}. All four must appear.
    for allowed in ("accepted", "dismissed", "snoozed", "expired"):
        assert allowed in msg


def test_cartesian_product_partition_is_exhaustive() -> None:
    """Sanity: legal + illegal covers every (from, to) pair exactly once."""
    legal = set(_legal_pairs())
    illegal = set(_illegal_pairs())
    full = set(itertools.product(_ALL_FROM_STATES, _ALL_TO_STATES))
    assert legal.isdisjoint(illegal)
    assert legal | illegal == full


def test_package_reexports_state_symbols() -> None:
    """``core.moment`` re-exports :func:`validate_transition` and :class:`IllegalTransition`."""
    from core import moment as pkg

    assert pkg.validate_transition is validate_transition
    assert pkg.IllegalTransition is IllegalTransition
