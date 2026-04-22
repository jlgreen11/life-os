"""Tests for :mod:`api.schemas`.

Strategy
--------
For each schema touched by the Week 8 skeleton we cover two things:

1. **Round-trip.** Build a realistic instance, dump it with
   ``model_dump()``, reload via ``model_validate()``, and assert the
   round-tripped object equals the original. This exercises both
   serialisation and deserialisation against the same shape.

2. **Rejection of malformed input.** Each schema is strict-mode
   (``extra="forbid"``), so passing an unknown field or omitting a
   required field must raise :class:`pydantic.ValidationError`. The
   tests spot-check representative failure modes rather than
   enumerating every field — Pydantic's own validation is trusted; we
   only confirm the contract is strict rather than permissive.

Enum values are taken from :mod:`core.moment.types` so if the
authoritative enum drifts, the tests break loudly rather than
silently diverging.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    ActionOut,
    ConnectorConfigIn,
    ConnectorOut,
    ContactDossierOut,
    ContactSummaryOut,
    DriftingContactOut,
    HealthOut,
    MetricsOut,
    MomentActionIn,
    MomentListOut,
    MomentOut,
    PeopleListOut,
    PersonaStyleOut,
    RoutineOut,
    StateHistoryOut,
    YouOut,
)
from core.moment.types import ActionKind, InsightType, MomentState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _moment_payload() -> dict:
    return {
        "id": "mom_01",
        "created_at": 1_700_000_000,
        "expires_at": 1_700_259_200,
        "insight": "You haven't texted Alex in 12 days; usual gap is 4.",
        "evidence": ["evt_1", "evt_2"],
        "evidence_hash": "deadbeef",
        "proposed_action": {
            "kind": ActionKind.DRAFT_MESSAGE.value,
            "params": {"to": "alex", "body": "hey!"},
        },
        "source_insight_type": InsightType.CADENCE.value,
        "state": MomentState.SUGGESTED.value,
        "scheduled_for": None,
        "context_trigger": "after_inactivity:4d:imessage",
        "snooze_until": None,
        "confidence": 0.72,
        "feedback_weight": 1.1,
        "state_history": [
            {
                "from_state": None,
                "to_state": MomentState.SUGGESTED.value,
                "ts": 1_700_000_000,
                "annotation": "producer:cadence",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Moment schemas
# ---------------------------------------------------------------------------


def test_moment_out_round_trip():
    payload = _moment_payload()
    moment = MomentOut.model_validate(payload)

    assert moment.id == "mom_01"
    assert moment.proposed_action.kind is ActionKind.DRAFT_MESSAGE
    assert moment.source_insight_type is InsightType.CADENCE
    assert moment.state is MomentState.SUGGESTED
    assert len(moment.state_history) == 1
    assert moment.state_history[0].to_state is MomentState.SUGGESTED

    dumped = moment.model_dump(mode="json")
    replay = MomentOut.model_validate(dumped)
    assert replay == moment


def test_moment_out_defaults_evidence_and_history_to_lists():
    payload = _moment_payload()
    payload.pop("evidence")
    payload.pop("state_history")

    moment = MomentOut.model_validate(payload)

    assert moment.evidence == []
    assert moment.state_history == []


def test_moment_out_rejects_unknown_field():
    payload = _moment_payload()
    payload["mood"] = "calm"  # mood never leaves the system

    with pytest.raises(ValidationError):
        MomentOut.model_validate(payload)


def test_moment_out_rejects_unknown_state():
    payload = _moment_payload()
    payload["state"] = "archived"  # not in the enum

    with pytest.raises(ValidationError):
        MomentOut.model_validate(payload)


def test_moment_out_rejects_missing_required_id():
    payload = _moment_payload()
    payload.pop("id")

    with pytest.raises(ValidationError):
        MomentOut.model_validate(payload)


def test_action_out_params_defaults_to_empty_dict():
    action = ActionOut.model_validate({"kind": ActionKind.NUDGE.value})

    assert action.params == {}
    assert action.kind is ActionKind.NUDGE


def test_state_history_from_state_nullable():
    entry = StateHistoryOut.model_validate({"from_state": None, "to_state": MomentState.SUGGESTED.value, "ts": 1})

    assert entry.from_state is None
    assert entry.annotation is None


def test_moment_list_out_buckets_default_to_empty_lists():
    payload = MomentListOut.model_validate({})

    assert payload.pending == []
    assert payload.scheduled == []
    assert payload.done == []


def test_moment_list_out_round_trip():
    m = MomentOut.model_validate(_moment_payload())
    m2 = m.model_copy(update={"id": "mom_02", "state": MomentState.DONE})

    listing = MomentListOut(pending=[m], scheduled=[], done=[m2])

    dumped = listing.model_dump(mode="json")
    replay = MomentListOut.model_validate(dumped)
    assert replay == listing
    assert replay.done[0].state is MomentState.DONE


def test_moment_action_in_all_optional():
    assert MomentActionIn.model_validate({}).snooze_until is None

    snooze = MomentActionIn.model_validate({"snooze_until": 1_700_000_000})
    assert snooze.snooze_until == 1_700_000_000

    edit = MomentActionIn.model_validate({"action_params": {"body": "edited"}})
    assert edit.action_params == {"body": "edited"}


def test_moment_action_in_rejects_unknown_field():
    with pytest.raises(ValidationError):
        MomentActionIn.model_validate({"force": True})


# ---------------------------------------------------------------------------
# You / People
# ---------------------------------------------------------------------------


def test_you_out_defaults_are_empty_containers():
    you = YouOut.model_validate({})

    assert you.observed_months == 0
    assert you.when_at_best == []
    assert you.how_you_write == []
    assert you.your_routines == []
    assert you.drifting == []


def test_you_out_round_trip_with_nested_lists():
    you = YouOut(
        observed_months=4,
        interactions_count=1200,
        confidence_pct=48,
        when_at_best=["mornings", "post-walk"],
        how_you_write=[
            PersonaStyleOut(audience="partner", tone="warm", formality=0.2, sample_size=180),
        ],
        your_routines=[
            RoutineOut(name="morning coffee", detected=True, confidence=0.8, sample_size=42),
            RoutineOut(name="weekly retro", detected=False, description="No routine detected yet"),
        ],
        drifting=[
            DriftingContactOut(contact_id="c1", name="Alex", days_since_last=12, usual_cadence_days=4),
        ],
    )

    dumped = you.model_dump(mode="json")
    replay = YouOut.model_validate(dumped)

    assert replay == you
    assert replay.your_routines[1].detected is False


def test_people_list_out_requires_you_field():
    with pytest.raises(ValidationError):
        PeopleListOut.model_validate({"needs_attention": [], "active_this_week": []})


def test_people_list_out_round_trip():
    listing = PeopleListOut(
        you=YouOut(),
        needs_attention=[
            ContactSummaryOut(
                contact_id="c1",
                name="Alex",
                last_contact_ts=1_699_000_000,
                cadence_deviation_days=8,
                needs_attention=True,
            ),
        ],
        active_this_week=[
            ContactSummaryOut(contact_id="c2", name="Jo", last_contact_ts=1_700_000_000),
        ],
        total=2,
        query="al",
    )

    dumped = listing.model_dump(mode="json")
    replay = PeopleListOut.model_validate(dumped)

    assert replay == listing
    assert replay.needs_attention[0].needs_attention is True


def test_contact_dossier_round_trip():
    dossier = ContactDossierOut(
        contact_id="c1",
        name="Alex",
        last_contact_ts=1_700_000_000,
        usual_cadence_days=4,
        comm_template="Hey {first_name}, how's your week going?",
        cadence_sparkline=[0, 1, 0, 2, 0, 0, 0],
        recent_topics=["climbing", "move"],
        predicted_next="Will reach out Saturday 6pm",
    )

    dumped = dossier.model_dump(mode="json")
    replay = ContactDossierOut.model_validate(dumped)

    assert replay == dossier


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------


def test_connector_out_round_trip_and_rejects_secret_fields():
    connector = ConnectorOut(
        id="proton_mail",
        kind="email",
        enabled=True,
        status="healthy",
        last_sync_at=1_700_000_000,
        last_error=None,
    )

    dumped = connector.model_dump(mode="json")
    replay = ConnectorOut.model_validate(dumped)
    assert replay == connector

    # Secrets must not be serialisable on ConnectorOut.
    with pytest.raises(ValidationError):
        ConnectorOut.model_validate({**connector.model_dump(), "password": "hunter2"})


def test_connector_config_in_accepts_partial_update():
    cfg = ConnectorConfigIn.model_validate({"enabled": False})
    assert cfg.enabled is False
    assert cfg.config is None
    assert cfg.secrets is None


def test_connector_config_in_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        ConnectorConfigIn.model_validate({"id": "proton_mail"})


# ---------------------------------------------------------------------------
# Health + Metrics
# ---------------------------------------------------------------------------


def test_health_out_round_trip_multi_key():
    health = HealthOut(
        ok=True,
        ts=1_700_000_000,
        connectors={"proton_mail": "healthy", "caldav": "degraded"},
        db_last_write_ts=1_699_999_500,
        scheduler_heartbeat_ts=1_699_999_900,
        producer_activity={"cadence": 12, "temporal": 4},
        pending_moments=17,
        notes=["scheduler within SLO"],
    )

    dumped = health.model_dump(mode="json")
    replay = HealthOut.model_validate(dumped)
    assert replay == health


def test_health_out_rejects_unknown_field():
    with pytest.raises(ValidationError):
        HealthOut.model_validate({"ok": True, "ts": 1, "overall": "healthy"})


def test_metrics_out_round_trip():
    metrics = MetricsOut(
        ts=1_700_000_000,
        counters={"moments_created_total": 42.0},
        gauges={"pending_moments": 17.0},
        histograms={
            "producer_latency_seconds": {"p50": 0.12, "p95": 0.42, "p99": 1.02},
        },
    )

    dumped = metrics.model_dump(mode="json")
    replay = MetricsOut.model_validate(dumped)
    assert replay == metrics
