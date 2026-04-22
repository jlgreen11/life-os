"""Moment state machine — legal transitions and validation.

This module holds the **authoritative** legal-transition table for the
Moment lifecycle. Every path that mutates ``Moment.state`` (the
``MomentRepository`` in ``storage/repos/moments.py``, the scheduler in
``core/moment/scheduler.py``, the API action endpoints in
``api/routes/now.py``, and any future dispatcher) must route its
proposed change through :func:`validate_transition` before writing to
SQLite. The repository itself wraps this call inside ``BEGIN
IMMEDIATE``, so an illegal transition rolls the whole transaction back.

References
----------
CEO plan § "State-machine transitions" (at
``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``)
defines the legal edges of this graph. If the CEO plan changes, this
table and the tests in ``tests/core/moment/test_state.py`` must change
together — the tests enumerate the full cartesian product of states
(including ``None``) and assert legal pairs match exactly.
"""

from __future__ import annotations

from core.moment.types import MomentState


class IllegalTransition(ValueError):
    """Raised when a caller attempts a transition not in the legal set.

    Subclasses :class:`ValueError` so existing repo-level error handling
    (which already catches ``ValueError`` around input validation)
    degrades gracefully when we tighten checks in newer layers.
    """


_LEGAL_TRANSITIONS: dict[MomentState | None, set[MomentState]] = {
    # Creation — a Moment may only be born into SUGGESTED.
    None: {MomentState.SUGGESTED},
    # SUGGESTED branches into four outcomes.
    MomentState.SUGGESTED: {
        MomentState.ACCEPTED,
        MomentState.DISMISSED,
        MomentState.SNOOZED,
        MomentState.EXPIRED,
    },
    # Once accepted, a Moment either completes or sits unfinished. The CEO
    # plan only names DONE as the onward edge; stalled acceptance is
    # modeled via the expires_at column, not a state transition.
    MomentState.ACCEPTED: {MomentState.DONE},
    # Snoozed Moments wake back into SUGGESTED or time out to EXPIRED.
    MomentState.SNOOZED: {MomentState.SUGGESTED, MomentState.EXPIRED},
    # Terminal states — no outbound transitions.
    MomentState.DISMISSED: set(),
    MomentState.DONE: set(),
    MomentState.EXPIRED: set(),
}


def validate_transition(
    from_state: MomentState | None,
    to_state: MomentState,
) -> None:
    """Raise :class:`IllegalTransition` unless ``from_state → to_state`` is legal.

    ``from_state`` is ``None`` for freshly created Moments (no prior
    persisted row). ``to_state`` is always a concrete :class:`MomentState`
    — creation without a state is not a modeled case.

    The function returns ``None`` on success so callers can use it as a
    guard expression immediately before a SQL ``UPDATE``.
    """
    allowed = _LEGAL_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise IllegalTransition(
            f"illegal transition {from_state!r} -> {to_state!r}; "
            f"legal targets: {sorted(s.value for s in allowed) or '(terminal)'}"
        )


__all__ = ["IllegalTransition", "validate_transition"]
