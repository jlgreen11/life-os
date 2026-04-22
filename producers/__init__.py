"""Life OS v2 — Moment producers.

Each producer in this package owns exactly one
:class:`core.moment.types.InsightType`. Producers are imported by
``main.py`` (Week 6) so their ``@register`` decorators populate the
:data:`core.moment.producer.PRODUCERS` registry on boot.

Phase 1 set (CEO plan § "Producers retained"):

- :class:`producers.cadence.CadenceProducer` — communication-cadence drift
- ``producers.relationship.RelationshipProducer`` — reciprocity drift (Week 4)
- ``producers.temporal.TemporalProducer`` — historical focus windows (Week 4)
- ``producers.spatial.SpatialProducer`` — place arrival/departure (Week 5)
- ``producers.comm_template.CommTemplateProducer`` — per-contact draft hints (Week 5)
- ``producers.routine.RoutineProducer`` — detected-routine reminders (Week 5)
"""

from producers.cadence import CADENCE_PRODUCER_KEY, CadenceProducer

__all__ = ["CADENCE_PRODUCER_KEY", "CadenceProducer"]
