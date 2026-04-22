"""Tests for :mod:`core.moment.producer`.

Covers the three behaviors called out in the Week 4 task body:

- **Producer is abstract and non-instantiable** — Instantiating
  :class:`Producer` directly, or any subclass that has not implemented
  :meth:`Producer.observe`, raises :class:`TypeError` from the
  standard :class:`abc.ABC` machinery.
- **Registry decorator works** — :func:`register` slots a concrete
  producer into :data:`PRODUCERS` keyed by its insight type, rejects
  double-registration, rejects non-Producer args, and refuses
  subclasses missing :attr:`insight_type`.
- **evidence_hash is deterministic across orderings** — Hashing the
  same evidence set under different iteration orders (and with
  duplicates) returns the same digest. Distinct sets return distinct
  digests. The output is stable hex suitable for the
  ``moments.evidence_hash`` column (length 64, [0-9a-f]).

Registration is global module state, so each test that mutates the
registry does so inside the ``clean_registry`` fixture that snapshots
and restores :data:`PRODUCERS` around the test body.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest

from core.moment.producer import (
    PRODUCERS,
    Producer,
    ProducerAlreadyRegistered,
    register,
)
from core.moment.types import InsightType, Moment


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot + restore the module-level PRODUCERS dict around a test.

    Tests that use ``@register`` or mutate :data:`PRODUCERS` directly
    leak global state otherwise, which makes test ordering matter and
    breaks parallel runs.
    """
    saved = dict(PRODUCERS)
    try:
        PRODUCERS.clear()
        yield
    finally:
        PRODUCERS.clear()
        PRODUCERS.update(saved)


# ---------------------------------------------------------------------------
# Producer is abstract
# ---------------------------------------------------------------------------


def test_producer_base_is_not_instantiable() -> None:
    """Instantiating ``Producer`` directly raises ``TypeError`` (abstract)."""
    with pytest.raises(TypeError):
        Producer()  # type: ignore[abstract]


def test_subclass_without_observe_is_not_instantiable(clean_registry: None) -> None:
    """A subclass that does not implement ``observe`` is still abstract."""

    class _Half(Producer):
        insight_type = InsightType.CADENCE

    with pytest.raises(TypeError):
        _Half()  # type: ignore[abstract]


def test_subclass_with_observe_is_instantiable(clean_registry: None) -> None:
    """Concrete subclass (with ``observe``) can be constructed."""

    class _Good(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return []

    instance = _Good()
    assert isinstance(instance, Producer)
    assert instance.insight_type is InsightType.CADENCE


def test_subclass_with_wrong_insight_type_attr_rejected_at_subclass_time() -> None:
    """A typo like ``insight_type = 'cadence'`` fails at class creation."""
    with pytest.raises(TypeError, match="insight_type must be an InsightType"):

        class _Bad(Producer):
            insight_type = "cadence"  # type: ignore[assignment]

            async def observe(self, event):
                return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_inserts_into_producers_mapping(clean_registry: None) -> None:
    """``@register`` stores the class under its ``insight_type``."""

    @register
    class _Cadence(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return []

    assert PRODUCERS[InsightType.CADENCE] is _Cadence


def test_register_returns_the_class_unchanged(clean_registry: None) -> None:
    """The decorator must be transparent — return the class identity."""

    class _Cadence(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return []

    assert register(_Cadence) is _Cadence


def test_register_rejects_double_registration(clean_registry: None) -> None:
    """Two producers cannot claim the same InsightType."""

    @register
    class _First(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return []

    with pytest.raises(ProducerAlreadyRegistered, match="CADENCE"):

        @register
        class _Second(Producer):
            insight_type = InsightType.CADENCE

            async def observe(self, event):
                return []


def test_register_is_idempotent_on_same_class(clean_registry: None) -> None:
    """Re-decorating the same class twice (e.g. import-cycle edge case) is a no-op."""

    class _Cadence(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return []

    register(_Cadence)
    # Second call with the *same* class should not raise.
    register(_Cadence)
    assert PRODUCERS[InsightType.CADENCE] is _Cadence


def test_register_rejects_non_producer(clean_registry: None) -> None:
    """``@register`` only accepts Producer subclasses."""

    class _NotAProducer:
        insight_type = InsightType.CADENCE

    with pytest.raises(TypeError, match="Producer subclass"):
        register(_NotAProducer)  # type: ignore[arg-type]


def test_register_rejects_bare_producer_base(clean_registry: None) -> None:
    """``@register`` cannot be applied to the abstract base itself."""
    with pytest.raises(TypeError, match="base class"):
        register(Producer)


def test_register_rejects_missing_insight_type(clean_registry: None) -> None:
    """A Producer subclass that forgot ``insight_type`` is rejected."""

    class _NoType(Producer):
        async def observe(self, event):
            return []

    with pytest.raises(TypeError, match="insight_type"):
        register(_NoType)


def test_registry_allows_distinct_insight_types(clean_registry: None) -> None:
    """Producers for different insight types coexist in the registry."""

    @register
    class _Cadence(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return []

    @register
    class _Relationship(Producer):
        insight_type = InsightType.RELATIONSHIP

        async def observe(self, event):
            return []

    assert set(PRODUCERS.keys()) == {InsightType.CADENCE, InsightType.RELATIONSHIP}
    assert PRODUCERS[InsightType.CADENCE] is _Cadence
    assert PRODUCERS[InsightType.RELATIONSHIP] is _Relationship


# ---------------------------------------------------------------------------
# evidence_hash
# ---------------------------------------------------------------------------


def test_evidence_hash_is_sha256_hex() -> None:
    """Digest is 64-char lowercase hex (SHA-256)."""
    digest = Producer.evidence_hash(["evt-1"])
    assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def test_evidence_hash_is_order_independent() -> None:
    """Different input orderings collapse to the same digest."""
    a = Producer.evidence_hash(["evt-1", "evt-2", "evt-3"])
    b = Producer.evidence_hash(["evt-3", "evt-2", "evt-1"])
    c = Producer.evidence_hash(["evt-2", "evt-1", "evt-3"])
    assert a == b == c


def test_evidence_hash_ignores_duplicates() -> None:
    """Duplicate event ids do not change the digest."""
    a = Producer.evidence_hash(["evt-1", "evt-2"])
    b = Producer.evidence_hash(["evt-1", "evt-2", "evt-1", "evt-2"])
    assert a == b


def test_evidence_hash_is_deterministic_across_calls() -> None:
    """Same input → same digest across repeated calls (no salt)."""
    events = ["a", "b", "c"]
    digests = {Producer.evidence_hash(events) for _ in range(5)}
    assert len(digests) == 1


def test_evidence_hash_distinct_sets_produce_distinct_digests() -> None:
    """Different evidence sets must not collide."""
    a = Producer.evidence_hash(["evt-1"])
    b = Producer.evidence_hash(["evt-2"])
    c = Producer.evidence_hash(["evt-1", "evt-2"])
    assert len({a, b, c}) == 3


def test_evidence_hash_handles_iterables_other_than_list() -> None:
    """Accepting ``Iterable`` means sets, generators, tuples all work."""
    base = Producer.evidence_hash(["a", "b"])
    assert Producer.evidence_hash(("a", "b")) == base
    assert Producer.evidence_hash({"a", "b"}) == base
    assert Producer.evidence_hash(iter(["a", "b"])) == base


def test_evidence_hash_empty_input_is_stable() -> None:
    """Empty evidence hashes to a stable sentinel digest (no crash)."""
    a = Producer.evidence_hash([])
    b = Producer.evidence_hash([])
    assert a == b
    assert re.fullmatch(r"[0-9a-f]{64}", a) is not None


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------


def test_package_reexports_producer_symbols() -> None:
    """``core.moment`` re-exports Producer, register, PRODUCERS, and errors."""
    from core import moment as pkg

    assert pkg.Producer is Producer
    assert pkg.register is register
    assert pkg.PRODUCERS is PRODUCERS
    assert pkg.ProducerAlreadyRegistered is ProducerAlreadyRegistered


# ---------------------------------------------------------------------------
# observe() contract sanity
# ---------------------------------------------------------------------------


def test_observe_signature_returns_list_of_moments(clean_registry: None) -> None:
    """Concrete producer returning Moments is accepted by the ABC contract."""
    import asyncio
    import uuid

    from core.moment.types import Action, ActionKind, InsightType

    class _Pass(Producer):
        insight_type = InsightType.CADENCE

        async def observe(self, event):
            return [
                Moment(
                    id=str(uuid.uuid4()),
                    created_at=event["timestamp"],
                    expires_at=event["timestamp"] + 3600,
                    insight="test",
                    evidence_hash=self.evidence_hash([event["id"]]),
                    proposed_action=Action(kind=ActionKind.NUDGE),
                    source_insight_type=InsightType.CADENCE,
                    evidence=[event["id"]],
                )
            ]

    p = _Pass()
    out = asyncio.run(p.observe({"id": "e1", "type": "email.received", "timestamp": 1_777_204_800}))
    assert len(out) == 1
    assert isinstance(out[0], Moment)
    assert out[0].source_insight_type is InsightType.CADENCE
