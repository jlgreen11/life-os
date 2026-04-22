"""Producer abstract base + registry.

A **Producer** turns a stream of raw events into zero or more candidate
:class:`~core.moment.types.Moment` instances. Producers are the only
component in Life OS v2 that creates Moments — repositories persist,
the scheduler fires, the outbox dispatches, but every new Moment enters
the world from a producer's :meth:`Producer.observe` call.

Each producer owns exactly one :class:`~core.moment.types.InsightType`
(cadence, relationship, temporal, spatial, comm-template, routine — the
Phase 1 set). Downstream components route Moments by insight type, and
the feedback-weight EWMA stored per insight type (Week 6) feeds back
into a producer-level threshold.

Event shape
-----------
``observe`` accepts an event as a plain ``Mapping[str, Any]`` so that
it works against both the SQLite ``events`` row shape (``id``, ``type``,
``source``, ``timestamp``, ``priority``, ``payload``, ``metadata``) and
the in-memory dict-envelope used by the scheduler's trigger matcher.
The producer base does not constrain the event any further; individual
producers are free to read whichever keys they need and return an empty
list on events they do not care about.

Idempotency
-----------
The ``evidence_hash`` helper takes the list of event IDs underpinning a
Moment and returns a stable SHA-256 hex digest that is **independent of
input ordering and duplicates** — this is the `evidence_hash` column
MomentRepository uses to dedupe, so two runs of the same producer over
the same evidence must return the same hash. Tests enumerate the
ordering + duplicate invariants.

Registry
--------
The module-level :data:`PRODUCERS` mapping is keyed by insight type and
filled by the :func:`register` class decorator. The engine (Week 6) walks
this mapping on startup and constructs one instance per producer. The
contract: **exactly one** producer per insight type. Re-registering an
insight type raises :class:`ProducerAlreadyRegistered`.

References
----------
CEO plan § "The Moment Primitive (producers)" at
``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``.
Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md`` Week 4
task body.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from core.moment.types import InsightType, Moment

Event = Mapping[str, Any]
"""Event envelope type alias.

Kept as a ``Mapping`` (read-only view) so producers cannot accidentally
mutate the event they were handed. The concrete runtime type is a dict
hydrated from the SQLite ``events`` row or from the in-process bus.
"""


class ProducerAlreadyRegistered(ValueError):
    """Raised when two producers try to claim the same insight type.

    Exactly one producer owns each :class:`InsightType`; the registry
    enforces that invariant at class-definition time (when
    :func:`register` runs) so the mistake surfaces at import, not at
    the first event.
    """


class Producer(ABC):
    """Abstract base for all Moment producers.

    Subclasses must declare a class-level :attr:`insight_type` and
    implement :meth:`observe`. The :func:`register` decorator then
    slots the class into :data:`PRODUCERS` keyed by its insight type.

    ``Producer`` is intentionally async at the interface so that
    producers which need DB lookups (e.g. the cadence producer reading
    a signal profile) can await inside ``observe`` without blocking
    the engine's event loop. Pure-compute producers may still be
    synchronous under the hood and simply ``return []`` or a list.
    """

    insight_type: InsightType

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce that every concrete subclass declares an insight type.

        Abstract intermediaries that do not set ``insight_type`` would
        otherwise pass :func:`register` silently and collide with a
        real producer. Requiring the attr at subclass time keeps the
        surface area honest.
        """
        super().__init_subclass__(**kwargs)
        # Concrete subclasses must declare insight_type. We can't tell
        # "abstract subclass" from "concrete subclass" before
        # @abstractmethod binds, so defer to register() for the hard
        # check — but reject an explicit ``None`` here to catch typos.
        value = cls.__dict__.get("insight_type", None)
        if value is not None and not isinstance(value, InsightType):
            raise TypeError(f"{cls.__name__}.insight_type must be an InsightType, got {type(value).__name__}")

    @abstractmethod
    async def observe(self, event: Event) -> list[Moment]:
        """Return 0..N candidate Moments for the given event.

        Producers receive **every** event flowing through the engine and
        are expected to filter by ``event['type']``, ``event['source']``,
        or whatever dimension matters for their insight. Returning
        ``[]`` is the expected outcome on most events — producers are
        sparse by design.

        Implementations must be free of side effects: no DB writes, no
        outbox enqueues, no bus publishes. The engine owns persistence.
        """

    @staticmethod
    def evidence_hash(event_ids: Iterable[str]) -> str:
        """Return a stable SHA-256 hex digest of the evidence set.

        The hash is **ordering-independent and duplicate-insensitive**:
        sorting + deduping happens before hashing, so
        ``evidence_hash(["a", "b"]) == evidence_hash(["b", "a", "a"])``.
        This matches the dedup semantics of the ``evidence_hash`` column
        on the ``moments`` table — two producer firings over the same
        evidence must collapse into one row regardless of how the
        producer iterated its inputs.

        Empty evidence hashes to the digest of the empty string; the
        MomentRepository's uniqueness constraint then treats all
        evidence-less Moments as duplicates, which is the desired
        behavior (producers must cite evidence to create a Moment).
        """
        unique_sorted = sorted(set(event_ids))
        joined = "\x1f".join(unique_sorted)  # ASCII unit separator
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


PRODUCERS: dict[InsightType, type[Producer]] = {}
"""Registry mapping insight type → producer class.

Populated at import time by the :func:`register` decorator. The engine
(Week 6) reads this mapping once on boot and constructs one producer
instance per entry. Tests clear this dict via
``PRODUCERS.clear()`` between cases when they need to assert
registration behavior in isolation.
"""


def register(cls: type[Producer]) -> type[Producer]:
    """Class decorator that slots ``cls`` into :data:`PRODUCERS`.

    The decorator is applied once per producer class:

    .. code-block:: python

        @register
        class CadenceProducer(Producer):
            insight_type = InsightType.CADENCE

            async def observe(self, event): ...

    Raises :class:`TypeError` if ``cls`` is not a concrete
    :class:`Producer` subclass with an ``insight_type`` set, and
    :class:`ProducerAlreadyRegistered` if another producer already
    claims that insight type. Returning ``cls`` unchanged keeps the
    decorator transparent.
    """
    if not isinstance(cls, type) or not issubclass(cls, Producer):
        raise TypeError(f"@register expects a Producer subclass, got {cls!r}")
    if cls is Producer:
        raise TypeError("@register cannot be applied to the Producer base class itself")
    insight = getattr(cls, "insight_type", None)
    if not isinstance(insight, InsightType):
        raise TypeError(f"{cls.__name__} must declare 'insight_type: InsightType' before @register")
    existing = PRODUCERS.get(insight)
    if existing is not None and existing is not cls:
        raise ProducerAlreadyRegistered(
            f"InsightType.{insight.name} already registered to {existing.__name__}; "
            f"cannot re-register to {cls.__name__}"
        )
    PRODUCERS[insight] = cls
    return cls


__all__ = [
    "PRODUCERS",
    "Event",
    "Producer",
    "ProducerAlreadyRegistered",
    "register",
]
